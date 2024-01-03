import math
import random
import torch
import fire
import os
import json
import numpy as np
import multiprocessing as mp
import torch.distributed as dist
import deepspeed
from datetime import datetime
from typing import Optional
from pprint import pprint, pformat
from torch.utils.data import DataLoader, IterableDataset

import sys

# sys.path.insert(0, "..")
sys.path.insert(0, ".")  # for submit to PAI
# put below `src.data` import above other `src.` to avoid `common_io` import error
from src.data import collator, vocab_builder, tokenizer, read_dataset
from src.models.graphgpt.configuration_graphgpt import GraphGPTConfig
from src.models.graphgpt.modeling_graphgpt import GraphGPTCausal
from src.utils import (
    loss_utils,
    optimization_utils,
    tokenizer_utils,
    modules_utils,
    misc_utils,
    print_trainable_parameters,
    print_params,
    inspect_tokenization_results,
    set_up_shuffle_and_sampler,
    worker_init_fn_seed,
)


def train(
    output_dir: str = "../exp/models/graph_llama_test",
    pretrain_cpt: str = "",
    data_dir: str = "../data/TUDataset",
    dataset_name: str = "reddit_threads",
    # tokenization config
    tokenization_config: str = "reddit_tokenization_config.json",
    attr_assignment: str = "",
    ignored_off: int = 0,
    add_eos: bool = False,
    # training config
    optimization_config: str = "",
    total_tokens: float = 2.56e10,
    warmup_tokens: float = 2.56e8,
    batch_size: int = 128,
    pad_to_multiple_of: int = 8,
    pack_tokens: float = 0,
    lr: float = 0.001,
    weight_decay: float = 0.1,
    eps: float = 1e-6,
    max_grad_norm: float = 1.0,
    logging_steps: int = 100,
    num_workers: int = 8,  # num of workers for data processing in DataLoader
    logit_adjust: int = 0,  # https://spaces.ac.cn/archives/7615
    # deepspeed config
    deepspeed_config: str = "",
    gradient_accumulation_steps: int = 1,
    local_rank: int = 0,
    # architecture config
    model_config: str = "",
    vocab_size: int = None,  # defaults to 32000
    hidden_size: int = 128,  # defaults to 4096
    num_hidden_layers: int = 2,  # defaults to 32
    # intermediate_size: int = 512,  # defaults to 11008
    # num_attention_heads: int = 4,  # defaults to 32
    hidden_act: str = "silu",  # defaults to "silu"
    max_position_embeddings: int = 128,  # defaults to 2048
    initializer_range: float = 0.02,  # defaults to 0.02
    tie_word_embeddings: int = 1,  # defaults to False
    # odps config
    tables: str = "",  # ODPS input table names
    outputs: str = "",  # ODPS output table names
    samples_per_saving: Optional[int] = None,
):
    (
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_hidden_layers,
    ) = modules_utils.set_up_model_architect(
        hidden_size=hidden_size, num_hidden_layers=num_hidden_layers
    )
    use_ddp = True if len(deepspeed_config) > 0 else False
    betas = (0.9, 0.95)
    min_lr = lr * 0.1  # from llama2 pre-train settings
    gpu_name = torch.cuda.get_device_name()
    if os.path.exists(os.path.join(output_dir, "log.csv")):
        print(
            f"log file {os.path.join(output_dir, 'log.csv')} exists, resume training from {output_dir} instead of initializing from pre-train ckp {pretrain_cpt}!"
        )
        pretrain_cpt = output_dir
    # 0. init distributed train and get gpu/device info
    if use_ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = os.environ.get("LOCAL_RANK")
    print(f"\nworld size: {world_size}, rank: {rank}, local rank: {local_rank}")
    rnd_seed = torch.random.initial_seed() - rank
    random.seed(rnd_seed)
    print(f"seed random with {rnd_seed}")
    steps_per_saving = samples_per_saving // (world_size * batch_size)
    print(f"\nsteps_per_saving: {steps_per_saving}")
    params = print_params(**locals())

    # 1. prepare data & tokenizer
    # 1.1 read configuration
    if len(pretrain_cpt) > 0:
        print(f"Ignoring input tokenization config file\n{tokenization_config}")
        tokenization_config = os.path.join(pretrain_cpt, "tokenization_config.json")
        print(f"Use saved pretrain tokenization config file\n{tokenization_config}\n")
    with open(tokenization_config, "r") as fp:
        tokenizer_config = json.load(fp)
        tokenizer_config["data_dir"] = data_dir
        tokenizer_config["name_or_path"] = (
            os.path.join(data_dir, tokenizer_config["name_or_path"])
            if len(pretrain_cpt) == 0
            else tokenizer_config["name_or_path"]
        )
        tokenizer_config["dataset"] = dataset_name
        tokenizer_config["task_type"] = "pretrain"
        tokenizer_config["semantics"]["attr_shuffle"] = True
        # TODO: remove in the future
    if len(attr_assignment) > 0:
        assert attr_assignment in tokenizer_utils.ATTR_ASSIGNMENT_TYPES
        tokenizer_config["semantics"]["attr_assignment"] = attr_assignment
    if ignored_off:
        # tokenizer_config["semantics"]["node"]["ignored_val"] = None
        tokenizer_config["semantics"]["edge"]["ignored_val"] = None
    pprint(tokenizer_config)
    # 1.2 get graph dataset
    dataset, raw_dataset = read_dataset(
        name=tokenizer_config["dataset"],
        # for local data file reading
        data_dir=data_dir,
        sampling_config=tokenizer_config["sampling"],
        # for odps data reading
        table=tables,
        mode="train",
        # general
        pretrain_mode=True,
        ensemble_datasets=tokenizer_config.get("ensemble_datasets", []),
    )
    if isinstance(dataset, IterableDataset):
        print("Iterable dataset, NOT printing elements")
    else:
        idx = dataset.sampler[0]
        print(dataset[idx])
    # 1.3 build vocab and then init tokenizer from the tokenization config
    vocab_builder.build_vocab(raw_dataset, tokenizer_config, rank)
    gtokenizer = tokenizer.GSTTokenizer(tokenizer_config, add_eos=add_eos)
    # 1.4 get train/test sampler
    train_dataset = dataset
    train_sampler = train_dataset.sampler
    random.shuffle(train_sampler)
    train_shuffle, train_sampler, train_cnt = set_up_shuffle_and_sampler(
        train_dataset, train_sampler
    )
    if pack_tokens > 0:
        gtokenizer.mpe = max_position_embeddings
        gtokenizer.dataset = train_dataset
        gtokenizer.sampler = tuple(train_sampler)
        gtokenizer.random_ratio = pack_tokens
        tokens_per_sample = max_position_embeddings
    else:
        tokens_per_sample = misc_utils.estimate_tokens_per_sample(
            gtokenizer, train_dataset, train_sampler, max_position_embeddings
        )
    inspect_tokenization_results(dataset, gtokenizer)
    total_num_steps = int(
        math.ceil(total_tokens / (tokens_per_sample * batch_size * world_size))
    )
    warmup_num_steps = int(
        math.ceil(warmup_tokens / (tokens_per_sample * batch_size * world_size))
    )
    epochs = int(
        math.ceil(total_tokens / (len(train_sampler) * tokens_per_sample * world_size))
    )
    print(
        f"\ntotal_num_steps: {total_num_steps}\nwarmup_num_steps: {warmup_num_steps}\nepochs: {epochs}\n"
    )

    # 2. set model
    # 2.1 init model config
    if len(model_config) > 0:
        with open(model_config, "r") as fp:
            model_config_dict = json.load(fp)
            print(
                f"Load model config\n{pformat(model_config_dict)}\nfrom\n{model_config}\n"
            )
    else:
        model_config_dict = {}
    if len(pretrain_cpt) > 0:
        model_config = os.path.join(pretrain_cpt, "config.json")
        print(f"Use saved pretrain model config file\n{model_config}\n")
        config = GraphGPTConfig().from_pretrained(model_config)
    else:
        config = GraphGPTConfig()
    config.update(
        {
            "vocab_size": gtokenizer.vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_attention_heads,
            "hidden_act": hidden_act,
            "max_position_embeddings": max_position_embeddings,
            "initializer_range": initializer_range,
            "tie_word_embeddings": bool(tie_word_embeddings),
            "bos_token_id": gtokenizer.get_bos_token_id(),
            "eos_token_id": gtokenizer.get_eos_token_id(),
        }
    )
    config.update(model_config_dict)
    print(config)
    # 2.2 create model
    if use_ddp:
        deepspeed.init_distributed(
            dist_backend="nccl", rank=rank, world_size=world_size
        )
    model = GraphGPTCausal(config)
    if use_ddp:
        # model.gradient_checkpointing_enable()  # this may cause problem in training
        model.model.gradient_checkpointing = True
        # model.model is Llama model, enable its gradient_checkpointing!
        # silence the warnings. Please re-enable for inference!
        model.config.use_cache = False
    else:
        model.to(device)
    print_trainable_parameters(model)
    # 2.21 load from ckp IF provided existing ckp and NOT resume from the ckp
    if (len(pretrain_cpt) > 0) and (pretrain_cpt != output_dir):
        ckp, _ = misc_utils.get_latest_ckp(pretrain_cpt)
        if use_ddp:
            print(f"Loading existing weights from ckp {ckp} using deepspeed API.")
            from deepspeed.utils.zero_to_fp32 import (
                get_fp32_state_dict_from_zero_checkpoint,
            )

            stat_dict = get_fp32_state_dict_from_zero_checkpoint(ckp)
            missing_keys, unexpected_keys = model.load_state_dict(
                stat_dict, strict=False
            )
            print(
                "load ckp using DeepSpeed API `get_fp32_state_dict_from_zero_checkpoint` and pytorch `load_state_dict`\n"
                f"missing keys: {missing_keys}\n"
                f"unexpected_keys: {unexpected_keys}\n"
            )
        else:
            print(f"Loading pretrained weights from ckp {ckp} using HF API.")
            model = model.from_pretrained(ckp, config=config)
        print("After loading weights from ckp:")
        print(model.config)
    print(model)
    # 2.3 Create optimizer (load optimization config if given)
    if len(optimization_config) > 0:
        with open(optimization_config, "r") as fp:
            opt_config_dict = json.load(fp)
            print(
                f"Load optimization config {pformat(opt_config_dict)} from {optimization_config}"
            )
    else:
        opt_config_dict = {}
    # 3.03 set optimizer and scheduler
    if use_ddp:
        with open(deepspeed_config, "r") as f:
            ds_config = json.load(f)
        train_batch_size = (
            int(os.environ["WORLD_SIZE"]) * batch_size * gradient_accumulation_steps
        )
        ds_config["train_micro_batch_size_per_gpu"] = batch_size
        ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps
        ds_config["train_batch_size"] = train_batch_size
        if max_grad_norm > 0:
            ds_config["gradient_clipping"] = max_grad_norm
        if "optimizer" in ds_config.keys():
            ds_config["optimizer"]["params"]["lr"] = lr
            ds_config["optimizer"]["params"]["betas"] = betas
            ds_config["optimizer"]["params"]["eps"] = eps
            ds_config["optimizer"]["params"]["weight_decay"] = weight_decay
            if "scheduler" in ds_config.keys():
                ds_config["scheduler"]["params"]["warmup_max_lr"] = lr
                ds_config["scheduler"]["params"]["warmup_min_lr"] = min_lr
                ds_config["scheduler"]["params"]["total_num_steps"] = total_num_steps
                ds_config["scheduler"]["params"]["warmup_num_steps"] = warmup_num_steps
                ds_config["scheduler"]["params"]["last_batch_iteration"] = -1
        if "zero_optimization" in ds_config.keys():
            if ds_config["zero_optimization"]["stage"] == 0:
                ds_config["zero_optimization"].pop("offload_optimizer")
        ds_config = optimization_utils.update_deepspeed_config(
            opt_config_dict, ds_config
        )
        if "tensorboard" in ds_config.keys():
            ds_config["tensorboard"]["output_path"] = output_dir
        print(f"\nds_config:\n{pformat(ds_config)}")
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
            mpu=None,
            dist_init_required=False,
        )
        device = model.device
    else:
        lr_scheduler = None
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
    # 2.4 Load model parameters and optimizer stats from ckp IF resuming from current ckp
    if (len(pretrain_cpt) > 0) and (pretrain_cpt == output_dir):
        ckp, _ = misc_utils.get_latest_ckp(pretrain_cpt)
        if use_ddp:
            print(
                f"Loading existing weights from ckp {ckp} using deepspeed API to resume training."
            )
            model.load_checkpoint(ckp)
            print("After loading weights from ckp:")
            print(model.__dict__["module"].config)
        else:
            print(
                f"Loading existing weights from ckp {ckp} using HF API to resume training."
            )
            model = model.from_pretrained(ckp, config=config)
            print("After loading weights from ckp:")
            print(model.config)
    if int(os.environ.get("RANK", 0)) == 0:
        if use_ddp:
            model.__dict__["module"].config.save_pretrained(output_dir)
        else:
            model.config.save_pretrained(output_dir)
        print(
            f"[{datetime.now()}] Finish -> Dump model config to `{output_dir}/config.json`"
        )
    print(f"[{datetime.now()}] Finish -> 2. set optimizer")
    # 3. set initial status
    # 3.01 set initial condition of optimization, either resuming from ckp or starting from scratch
    if pretrain_cpt == output_dir:
        _, prev_epoch = misc_utils.get_latest_ckp(pretrain_cpt)
        last_step_index = prev_epoch * steps_per_saving
        ls_log, _, _ = misc_utils.load_all(output_dir)
        last_step_index_from_log = int(ls_log[-1].strip().split(",")[2])
        assert (
            last_step_index >= last_step_index_from_log
        ), f"last_step_index: {last_step_index}, last_step_index_from_log: {last_step_index_from_log}"
        print(
            f"Resume training from {pretrain_cpt} with last_step_index {last_step_index}!"
        )
        ep_init = prev_epoch
        j_init = last_step_index
    else:
        last_step_index = -1
        ep_init = 0
        j_init = 0
        ls_log = ["epoch,lr,local_step,global_step,train_loss\n"]
    # 3.1 init collator
    collator_fn = collator.DataCollatorForGSTCausal(
        tokenizer=gtokenizer,
        max_length=max_position_embeddings,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    )
    print(f"[{datetime.now()}] Finish -> 3.1 init collator")
    # # 3.2 set-up loader
    # print(f"[{datetime.now()}] Finish -> 3.2 set-up loader")
    if int(os.environ.get("RANK", 0)) == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, "tokenization_config.json"), "w+") as fp:
            json.dump(tokenizer_config, fp, indent=4)
        print(f"[{datetime.now()}] Finish -> Dump to `tokenization_config.json`")
        with open(os.path.join(output_dir, "params.txt"), "w+") as fp:
            fp.write(params)
        print(f"[{datetime.now()}] Finish -> Dump to `params.txt`")
        if use_ddp:
            with open(os.path.join(output_dir, "ds_config.json"), "w+") as fp:
                json.dump(model.config, fp, indent=4)
            print(f"[{datetime.now()}] Finish -> Dump to `ds_config.json`")

    if logit_adjust:  # https://spaces.ac.cn/archives/7615
        shape = (config.vocab_size,)
        prior = np.zeros(shape, dtype=np.int64)
        print(f"Applying logit adjustment to the loss with initial prior\n{prior}")
    else:
        prior = None

    print(f"[{datetime.now()}] Training start ...")
    j = j_init
    ep = ep_init
    for epoch in range(epochs):
        model.train()
        print(f"Re-initialize train-loader with shuffled sampler and reset dataset!")
        train_dataset.reset_samples(epoch)
        train_sampler = train_dataset.sampler
        random.shuffle(train_sampler)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collator_fn,
            worker_init_fn=worker_init_fn_seed,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
        )
        # print(f"Top 10 samples' idx:\n{train_loader.sampler[:10]}")
        t_start = datetime.now()
        for i, data in enumerate(train_loader):
            # Iterate in batches over the training dataset.
            # print(data["input_ids"].is_pinned())
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"].to(device)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                prior=loss_utils.convert_prior_cnt_to_logits(
                    prior, len(data["input_ids"].shape), device
                ),
            )  # Perform a single forward pass.
            loss = output.loss
            if use_ddp:
                model.backward(loss)  # Derive gradients.
                model.step()
            else:
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.
            if logit_adjust:  # https://spaces.ac.cn/archives/7615
                prior = loss_utils.update_logits_prior(
                    prior, data["labels"], (config.vocab_size,)
                )
            if j % logging_steps == 0:
                t_interval = (datetime.now() - t_start).total_seconds()
                samples_per_second = round(i * batch_size / t_interval, 1)
                tokens_per_second = round(
                    i * batch_size * tokens_per_sample / t_interval
                )
                print(
                    f"[{datetime.now()}][epoch {ep}][local {epoch}: {i}][global {j}] train_loss: {round(loss.item(),7)}, {samples_per_second} samples / {tokens_per_second} tokens per sec"
                )
                # Reduce SUM to get the loss from all the GPUs to RANK=0
                # refer: https://github.com/microsoft/DeepSpeed/discussions/2377#discussioncomment-3765282
                dist.reduce(loss, 0)
                loss = loss / world_size
                curr_lr = lr_scheduler.get_lr() if lr_scheduler is not None else [lr]
                ls_log.append(f"{ep},{curr_lr[0]},{i},{j},{loss}\n")
            if j == total_num_steps:
                print(
                    f"Total number of steps {total_num_steps} reached, break inner loop!!!"
                )
                break

            if (j % steps_per_saving == 0) and (j > j_init):
                ep += 1
                print(
                    f"[{datetime.now()}][end of epoch {ep}][local {epoch}: {i}][global {j}] Trained with {j*tokens_per_sample*batch_size*world_size} tokens! Saving ckp and logs!"
                )
                print(f"[{datetime.now()}][input_id] shape: {input_ids.shape}")
                misc_utils.save_ckp(output_dir, model, ep, use_ddp)
                if int(os.environ.get("RANK", 0)) == 0:
                    # save ckp and logs
                    misc_utils.save_all(
                        output_dir, model, ep, save_model=False, ls_log=ls_log
                    )
                if logit_adjust:
                    print(
                        f"[{datetime.now()}][prior]{prior/prior.sum()}\n{loss_utils.convert_prior_cnt_to_logits(prior, len(data['input_ids'].shape), device)}"
                    )

            j += 1
        if j == total_num_steps:
            print(
                f"Total number of steps {total_num_steps} reached, break outer loop!!!"
            )
            break
    ep += 1
    print(
        f"[{datetime.now()}][end of training][epoch {ep}][local {epoch}: {i}][global {j}] Trained with {j*tokens_per_sample*batch_size*world_size} tokens! Saving ckp and logs!"
    )
    misc_utils.save_ckp(output_dir, model, ep, use_ddp)
    if int(os.environ.get("RANK", 0)) == 0:
        # save ckp and logs
        misc_utils.save_all(output_dir, model, ep, save_model=False, ls_log=ls_log)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    fire.Fire(train)
