import copy
import random
import torch
import fire
import os
import json
import multiprocessing as mp
import torch.distributed as dist
import deepspeed
from datetime import datetime
from typing import Optional, List
from pprint import pprint, pformat
from torch.utils.data import DataLoader, IterableDataset

import sys

# sys.path.insert(0, "..")
sys.path.insert(0, ".")  # for submit to PAI
# put below `src.data` import above other `src.` to avoid `common_io` import error
from src.data import collator, vocab_builder, tokenizer, read_dataset
from src.models.graphgpt.configuration_graphgpt import GraphGPTConfig
from src.models.graphgpt.modeling_graphgpt import (
    GraphGPTForSequenceClassification,
)
from src.utils import (
    optimization_utils,
    tokenizer_utils,
    loader_utils,
    modules_utils,
    misc_utils,
    loss_utils,
    print_trainable_parameters,
    print_params,
    inspect_tokenization_results,
    get_metrics,
    evaluate_ogb,
    format_ogb_output_for_csv,
    set_up_shuffle_and_sampler,
    worker_init_fn_seed,
)


def _get_pos_weight(graph, train_idx):
    pos_cnt_per_label = graph.y[train_idx].sum(axis=0)
    pos_weight = (graph.y[train_idx].shape[0] - pos_cnt_per_label) / pos_cnt_per_label
    return pos_weight


@torch.no_grad()
def evaluate(
    model,
    problem_type,
    device,
    num_labels,
    loader,
    task_level,
    dataset_name,
    tensor_shape_list,
):
    model.eval()
    cls_metrics = get_metrics(problem_type, device, num_labels=num_labels)
    test_loss = 0
    for j, test_data in enumerate(loader, 1):
        input_ids = test_data["input_ids"].to(device)
        attention_mask = test_data["attention_mask"].to(device)
        task_labels = test_data[f"{task_level}_labels"].to(device)
        task_labels = (
            task_labels.float()
            if problem_type == "multi_label_classification"
            else task_labels
        )
        res = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=task_labels,
        )  # Perform a single forward pass.
        # record loss
        test_loss += res.loss
        # record metrics
        idx = test_data["idx"].to(device)
        cls_metrics.update(res.logits, task_labels, idx)
    test_loss = test_loss / j
    cls_metrics.compute()
    ogb_input_dict = cls_metrics.to_dict()
    if len(tensor_shape_list) > 1:
        # gather results from different gpus if there are more than 1 gpu
        for key, val in ogb_input_dict.items():
            tensor_list = [
                torch.zeros(
                    cls_metrics.get_output_shape(shape, key),
                    dtype=val.dtype,
                    device=device,
                )
                for shape in tensor_shape_list
            ]
            dist.all_gather(tensor_list, val.to(device))
            tensor_out = torch.cat(tensor_list)
            ogb_input_dict[key] = tensor_out
            cnt_nans = torch.isnan(tensor_out).sum().item()
            print(
                f"[{datetime.now()}] Update ogb_input_dict element {key} with val of shape {tensor_out.shape} of type {tensor_out.dtype} of NANs {cnt_nans}"
            )
    ogb_eval_res = evaluate_ogb(dataset_name, ogb_input_dict)
    return test_loss, cls_metrics, ogb_eval_res, ogb_input_dict


def train(
    output_dir: str = "../exp/models/graph_llama_classification_test",
    pretrain_cpt: str = "",
    data_dir: str = "../data/TUDataset",
    dataset_name: str = "reddit_threads",
    with_prob: int = 0,  # sample graphs for train proportional to their number of eulerian paths/num_nodes
    save_pred: int = 0,  # whether to save prediction results
    # tokenization config
    tokenization_config: str = "reddit_tokenization_config.json",
    attr_assignment: str = "",
    attr_shuffle: int = 1,
    ignored_off: int = 0,
    # training config
    optimization_config: str = "",
    epochs: int = 1,
    warmup_epochs: float = 0.25,
    batch_size: int = 128,
    pad_to_multiple_of: int = 8,
    lr: float = 0.0001,
    eps: float = 1e-8,
    betas: List[float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    logging_steps: int = 100,
    num_workers: int = 8,  # num of workers for data processing in DataLoader
    freeze: int = -1,  # how to freeze the params of backbone architecture: -1->no, 0->embedding
    # eval config
    eval_only: int = 0,
    k_samplers: int = 262144,  # 2^14=16384  2^16=65536  2^18=262144
    true_valid: int = 0,
    # deepspeed config
    deepspeed_config: str = "",
    gradient_accumulation_steps: int = 1,
    local_rank: int = 0,
    # architecture cofig
    model_config: str = "",
    vocab_size: int = None,  # defaults to 32000
    hidden_size: int = 128,  # defaults to 4096
    num_hidden_layers: int = 2,  # defaults to 32
    # intermediate_size: int = 512,  # defaults to 11008
    # num_attention_heads: int = 4,  # defaults to 32
    hidden_act: str = "silu",  # defaults to "silu"
    max_position_embeddings: int = 128,  # defaults to 2048
    initializer_range: float = 0.02,  # defaults to 0.02
    # supervised task config
    num_labels: Optional[int] = 2,
    mlp: Optional[List[int]] = None,
    dropout: float = 0,  # dropout for mlp layers
    problem_type: Optional[
        str
    ] = None,  # single_label_classification|multi_label_classification|regression
    loss_type: Optional[str] = "",  # auc
    task_level: Optional[str] = "graph",  # pretrain|graph|edge|node
    # odps config
    tables: str = "",  # ODPS input table names
    outputs: str = "",  # ODPS output table names
    # others
    samples_per_eval: int = 0,
    seed: int = 0,
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
    min_lr = 0 * lr
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
    eval_steps = samples_per_eval // (world_size * batch_size)
    params = print_params(**locals())

    # 1. prepare data & tokenizer
    # 1.1 set-up tokenization config
    with open(tokenization_config, "r") as fp:
        tokenizer_config = json.load(fp)
        sampling_config = tokenizer_config["sampling"]
        print(
            f"Use sampling config {sampling_config} for {dataset_name} from {tokenization_config}"
        )
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
        tokenizer_config["task_type"] = task_level
        tokenizer_config["sampling"] = sampling_config
        tokenizer_config["pretrain_cpt"] = pretrain_cpt
        tokenizer_config["semantics"]["graph"] = {
            "discrete": None,
            "continuous": None,
            "ignored_val": None,
        }
        tokenizer_config["semantics"]["add_3d"] = False
        tokenizer_config["semantics"]["attr_shuffle"] = bool(attr_shuffle)
    if len(attr_assignment) > 0:
        assert attr_assignment in tokenizer_utils.ATTR_ASSIGNMENT_TYPES
        tokenizer_config["semantics"]["attr_assignment"] = attr_assignment
    # if ignored_off:
    tokenizer_config["semantics"]["node"]["ignored_val"] = "0"
    tokenizer_config["semantics"]["edge"]["ignored_val"] = "0"
    pprint(tokenizer_config)
    # 1.2 get graph dataset
    train_dataset, valid_dataset, test_dataset, raw_dataset = read_dataset(
        name=dataset_name,
        # for local graph data file reading
        data_dir=data_dir,
        sampling_config=tokenizer_config["sampling"],
        # general
        pretrain_mode=False,
        return_valid_test=True,
        with_prob=bool(with_prob),
        # for odps data reading
        table=tables,
        mode="all",
        supervised_task=task_level,
        true_valid=true_valid,
    )
    for dataset in [train_dataset, valid_dataset, test_dataset]:
        if isinstance(dataset, IterableDataset):
            print("Iterable dataset, NOT printing elements")
        else:
            idx = dataset.sampler[0]
            print(dataset[idx])
    # 1.3 build vocab and then init tokenizer from the tokenization config
    vocab_builder.build_vocab(raw_dataset, tokenizer_config, rank)
    gtokenizer = tokenizer.GSTTokenizer(tokenizer_config)
    inspect_tokenization_results(train_dataset, gtokenizer)
    # 1.4 get train/test sampler
    train_sampler = train_dataset.sampler
    random.shuffle(train_sampler)
    train_shuffle, train_sampler, train_cnt = set_up_shuffle_and_sampler(
        train_dataset, train_sampler
    )
    train_sampler_for_eval = random.sample(
        train_sampler, min(len(train_sampler), k_samplers)
    )
    steps_per_epoch = (len(train_sampler) // world_size) // batch_size
    print(f"steps_per_epoch: {steps_per_epoch}")
    # due to `drop_last` in train_loader, use //
    samples_per_gpu = len(train_sampler) // world_size
    total_num_steps = epochs * (samples_per_gpu // batch_size)
    warmup_num_steps = int(warmup_epochs * (samples_per_gpu // batch_size))
    print(
        f"\ntotal_num_steps: {total_num_steps}\nwarmup_num_steps: {warmup_num_steps}\n"
    )

    valid_sampler = (
        valid_dataset.sampler
        if hasattr(valid_dataset, "sampler")
        else list(range(len(valid_dataset)))
    )
    random.shuffle(valid_sampler)
    valid_shuffle, valid_sampler, valid_cnt = set_up_shuffle_and_sampler(
        valid_dataset, valid_sampler
    )
    tensor_shape = len(valid_sampler) // world_size
    valid_tensor_shape_list = [tensor_shape] * world_size
    valid_sampler = loader_utils.distribute_sampler(valid_sampler, world_size, rank)[
        :tensor_shape
    ]
    valid_sampler_for_eval = random.sample(
        valid_sampler, min(len(valid_sampler), k_samplers)
    )

    test_sampler = (
        test_dataset.sampler
        if hasattr(test_dataset, "sampler")
        else list(range(len(test_dataset)))
    )
    random.shuffle(test_sampler)
    test_shuffle, test_sampler, test_cnt = set_up_shuffle_and_sampler(
        test_dataset, test_sampler
    )
    tensor_shape = len(test_sampler) // world_size
    test_tensor_shape_list = [tensor_shape] * world_size
    test_sampler = loader_utils.distribute_sampler(test_sampler, world_size, rank)[
        :tensor_shape
    ]
    test_sampler_for_eval = random.sample(
        test_sampler, min(len(test_sampler), k_samplers)
    )

    # 2. set model
    # 2.1 init model config
    if len(model_config) > 0:
        with open(model_config, "r") as fp:
            model_config_dict = json.load(fp)
            print(f"Load model config {pformat(model_config_dict)} from {model_config}")
    else:
        model_config_dict = {}
    if len(pretrain_cpt) > 0:
        model_config = os.path.join(pretrain_cpt, "config.json")
        print(f"Use saved pretrain model config file\n{model_config}\n")
        config = GraphGPTConfig().from_pretrained(model_config)
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
    else:
        config = GraphGPTConfig()
        num_key_value_heads = num_attention_heads
    config.update(
        {
            "vocab_size": gtokenizer.vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "hidden_act": hidden_act,
            "max_position_embeddings": max_position_embeddings,
            "initializer_range": initializer_range,
            "bos_token_id": gtokenizer.get_bos_token_id(),
            "eos_token_id": gtokenizer.get_eos_token_id(),
            "cls_token_id": gtokenizer.get_gsum_token_id(),
            "num_labels": num_labels,
            "mlp": mlp,
            "dropout": dropout,
            "problem_type": problem_type,
            "loss_type": loss_type if len(loss_type) > 0 else None,
            "num_neg": loss_utils.get_neg_ratio(tokenizer_config["sampling"]),
        }
    )
    config.update(model_config_dict)
    print(config)
    # 2.2 create model
    if use_ddp:
        deepspeed.init_distributed(
            dist_backend="nccl", rank=rank, world_size=world_size
        )
    model = GraphGPTForSequenceClassification(config)
    # model.pos_weight = _get_pos_weight(raw_dataset[0], train_idx).to(device)
    if use_ddp:
        # model.gradient_checkpointing_enable()
        # will this affect the final results???
        model.model.gradient_checkpointing = True
        # enable gradient_checkpointing for Llama backbone
        # silence the warnings. Please re-enable for inference!
        model.config.use_cache = False
    else:
        model.to(device)
    if freeze > -1:  # 0->freeze embedding; 1->embed+1st layer
        modules_utils.freeze_llama_layers(model, freeze)
    print_trainable_parameters(model)
    # 2.21 load from ckp IF provided existing ckp and NOT resume from the ckp
    if (len(pretrain_cpt) > 0) and (pretrain_cpt != output_dir):
        ckp, prev_epoch = misc_utils.get_latest_ckp(pretrain_cpt)
        print(f"Loading pretrained weights from ckp {ckp}")
        try:
            model = model.from_pretrained(ckp, config=config)
            print(f"load ckp using HF API `model.from_pretrained`")
        except Exception as inst:
            # print(type(inst))
            # print(inst.args)
            print(inst)
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
        print(
            f"After loading weights from ckp:\n{model.config}\nnum_labels: {model.num_labels}\nmodel-type: {model.dtype}\n\n{model}"
        )
    # 2.3 Create optimizer (load optimization config if given)
    if len(optimization_config) > 0:
        with open(optimization_config, "r") as fp:
            opt_config_dict = json.load(fp)
            print(
                f"Load optimization config {pformat(opt_config_dict)} from {optimization_config}"
            )
    else:
        opt_config_dict = {}
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
            ds_config = optimization_utils.update_deepspeed_config(
                opt_config_dict, ds_config
            )  # overwrite some ds config with input opt config
            if ("scheduler" in ds_config.keys()) and (
                ds_config["scheduler"]["type"] in loss_utils.DS_SCHEDULER_LS
            ):
                ds_config["scheduler"]["params"] = loss_utils.set_ds_scheduler(
                    ds_config["scheduler"]["type"],
                    ds_config["scheduler"]["params"],
                    # for WarmupLR & WarmupDecayLR
                    warmup_max_lr=lr,
                    warmup_min_lr=min_lr,
                    warmup_num_steps=warmup_num_steps,
                    total_num_steps=total_num_steps,
                    # for LRRangeTest
                    lr_range_test_min_lr=min_lr,
                    lr_range_test_step_size=warmup_num_steps,
                    # for OneCycle
                    cycle_min_lr=min_lr,
                    cycle_max_lr=lr,
                    cycle_first_step_size=warmup_num_steps,
                )
                ds_config["scheduler"]["params"]["last_batch_iteration"] = -1
        if "zero_optimization" in ds_config.keys():
            if ds_config["zero_optimization"]["stage"] == 0:
                ds_config["zero_optimization"].pop("offload_optimizer")
        ds_config["flops_profiler"]["enabled"] = False
        if "tensorboard" in ds_config.keys():
            ds_config["tensorboard"]["output_path"] = output_dir
        print(f"\nds_config:\n{pformat(ds_config)}")
        if ("scheduler" in ds_config.keys()) and (
            ds_config["scheduler"]["type"] not in loss_utils.DS_SCHEDULER_LS
        ):
            non_ds_scheduler, scheduler_conf = loss_utils.set_py_scheduler(
                ds_config["scheduler"]["type"],
                ds_config,
                # for CyclicLR
                base_lr=min_lr,
                max_lr=lr,
                step_size_up=warmup_num_steps,
                # for CosineAnnealingLR
                T_max=total_num_steps,
                eta_min=min_lr,
                # for CosineAnnealingWarmRestarts
                T_0=warmup_num_steps,
                # for OneCycleLR
                total_steps=total_num_steps,
                pct_start=warmup_num_steps / total_num_steps,
                min_lr=min_lr,
                # for general
                last_step_index=-1,
            )
        else:
            non_ds_scheduler, scheduler_conf = None, {}
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            lr_scheduler=non_ds_scheduler,
            config=ds_config,
            mpu=None,
            dist_init_required=False,
        )
        device = model.device
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        lr_scheduler = None
    # 2.4 Load model parameters and optimizer stats from ckp IF resuming from current ckp
    if (len(pretrain_cpt) > 0) and (pretrain_cpt == output_dir):
        ckp, prev_epoch = misc_utils.get_latest_ckp(pretrain_cpt)
        print(f"Loading pretrained weights from ckp {ckp}")
        model.load_checkpoint(ckp)
        print(
            f"After loading weights from ckp:\n{model.__dict__['module'].config}\nnum_labels: {model.num_labels}\nmodel-type: {model.dtype}\n\n{model}"
        )
        # model.to(dtype=torch.float32)
        # print(f"converting loaded model to {model.dtype}")
        # seems useless. Even if model saved in fp16, after loading with `from_pretrained`, it becomes fp32
    if int(os.environ.get("RANK", 0)) == 0:
        model.__dict__["module"].config.save_pretrained(output_dir)
    # 3. set initial status
    # 3.01 set initial condition of optimization, either resuming from ckp or starting from scratch
    if pretrain_cpt == output_dir:
        last_step_index = (prev_epoch + 1) * steps_per_epoch
        ls_log, ls_result, ls_loss = misc_utils.load_all(
            output_dir, load_log=True, load_result=True, load_loss=True
        )

        last_step_index_from_log = int(ls_log[-1].strip().split(",")[2])
        assert (
            last_step_index >= last_step_index_from_log
        ), f"last_step_index: {last_step_index}, last_step_index_from_log: {last_step_index_from_log}"

        last_step_index_from_result = int(ls_result[-1].strip().split(",")[1])
        assert (
            last_step_index == last_step_index_from_result
        ), f"last_step_index: {last_step_index}, last_step_index_from_result: {last_step_index_from_result}"

        last_step_index_from_loss = int(ls_loss[-1].strip().split(",")[2])
        assert (
            last_step_index >= last_step_index_from_loss
        ), f"last_step_index: {last_step_index}, last_step_index_from_loss: {last_step_index_from_loss}"

        print(
            f"Resume training from {pretrain_cpt} with last_step_index {last_step_index}!"
        )
        ep_init = prev_epoch + 1
        j_init = last_step_index
    else:
        last_step_index = -1
        ep_init = 0
        j_init = 0
        ls_log = ["epoch,local_step,global_step,train_loss,test_loss,metrics\n"]
        ls_result = [
            "epoch,global_step,all_lrs,lr,train_metric,valid_metric,test_metrics\n"
        ]
        ls_loss = ["epoch,local_step,global_step,train_loss\n"]
    # if last_step_index == total_num_steps:
    #     print(
    #         f"[WARNING] last_step_index == total_num_steps == {last_step_index}! SET to -1! lr RESTARTED!"
    #     )
    #     last_step_index = -1
    # if got `KeyError: "param 'initial_lr' is not specified`
    # refer: https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822/5

    # 3.1 init collator
    collator_fn = collator.DataCollatorForGSTCausal(
        tokenizer=gtokenizer,
        max_length=max_position_embeddings,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    )
    batch_size_eval = 16  # small to avoid OOM when evaluating
    train_loader_for_eval = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_eval,
        sampler=train_sampler_for_eval,
        num_workers=num_workers,
        collate_fn=collator_fn,
        worker_init_fn=worker_init_fn_seed,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size_eval,
        sampler=valid_sampler,
        num_workers=num_workers,
        collate_fn=collator_fn,
        worker_init_fn=worker_init_fn_seed,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_eval,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=collator_fn,
        worker_init_fn=worker_init_fn_seed,
        pin_memory=True,
    )
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
            tmp_ds_config = copy.deepcopy(model.config)
            tmp_ds_config.update(scheduler_conf)
            with open(os.path.join(output_dir, "ds_config.json"), "w+") as fp:
                json.dump(tmp_ds_config, fp, indent=4)
            print(f"[{datetime.now()}] Finish -> Dump to `ds_config.json`")

    enable_fp16 = False if ds_config.get("bf16", {}).get("enabled", False) else True
    print(f"enable_fp16: {enable_fp16}")
    i = 0
    j = j_init
    print(
        f"[{datetime.now()}] Training start with j_init {j_init} and ep_init {ep_init} ..."
    )
    for epoch in range(ep_init, epochs):
        if not eval_only:
            model.train()
            print(
                f"Re-initialize train-loader with shuffled sampler and reset dataset!"
            )
            train_dataset.reset_samples(epoch) if hasattr(
                train_dataset, "reset_samples"
            ) else None
            # train_sampler = (
            #     train_dataset.sampler
            #     if hasattr(train_dataset, "sampler")
            #     else train_sampler
            # ) TODO: make below compatible with other datasets
            # random.shuffle(train_sampler)
            train_sampler = loader_utils.distribute_sampler_with_rnd_seed(
                train_dataset.sample_idx, world_size, rank, seed=seed + epoch
            )
            # train_sampler = misc_utils.special_sort(train_sampler)
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=train_sampler,
                num_workers=num_workers,
                collate_fn=collator_fn,
                worker_init_fn=worker_init_fn_seed,
                pin_memory=True,
                prefetch_factor=2,
                drop_last=True,
            )
            # print(f"Top 10 samples' idx:\n{train_loader.sampler[:10]}")
            t_start = datetime.now()
            for i, data in enumerate(train_loader):
                # Iterate in batches over the training dataset.
                print(
                    f"[sample idx top 10][local i:{i}]{data['idx'][:10]}"
                ) if i == 0 else None
                input_ids = data["input_ids"].to(device)
                attention_mask = data["attention_mask"].to(device)
                task_labels = data[f"{task_level}_labels"].to(device)
                task_labels = (
                    task_labels.float()
                    if problem_type == "multi_label_classification"
                    else task_labels
                )
                if enable_fp16 == True and problem_type == "regression":
                    # task_labels = task_labels.round(decimals=5)  # avoid overfitting?
                    task_labels = task_labels.half()  # avoid overfitting?

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=task_labels,
                )  # Perform a single forward pass.
                loss = output.loss
                if use_ddp:
                    model.backward(loss)  # Derive gradients.
                    model.step()
                else:
                    loss.backward()  # Derive gradients.
                    optimizer.step()  # Update parameters based on gradients.
                    optimizer.zero_grad()  # Clear gradients.
                if j % logging_steps == 0:
                    t_interval = (datetime.now() - t_start).total_seconds()
                    samples_per_second = round(i * batch_size / t_interval)
                    print(
                        f"[{datetime.now()}][epoch {epoch}][local {i}][global {j}] train_loss: {loss}, processed {samples_per_second} samples per second!"
                    )
                    # Reduce SUM to get the loss from all the GPUs to RANK=0
                    # refer: https://github.com/microsoft/DeepSpeed/discussions/2377#discussioncomment-3765282
                    dist.reduce(loss, 0)
                    loss = loss / world_size
                    ls_loss.append(f"{epoch},{i},{j},{loss}\n")
                j += 1
        else:
            ckp = os.path.join(pretrain_cpt, f"epoch_{epoch}")
            if os.path.exists(ckp):
                print(f"Loading weights from ckp {ckp} for evaluation")
                model = model.from_pretrained(ckp, config=config).to(device)
            else:
                print(f"ckp {ckp} doesn't exists, skip it!")
        print(
            f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Evaluate on partial train data {k_samplers*world_size} -> {k_samplers}!"
        )
        tr_loss, tr_cls_metrics, tr_ogb_eval_res, tr_triplet = evaluate(
            model,
            problem_type,
            device,
            num_labels,
            train_loader_for_eval,
            task_level,
            tokenizer_config["dataset"],
            tensor_shape_list=[k_samplers] * world_size,
        )
        print(
            f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Evaluate on full valid data {valid_cnt} -> {len(valid_sampler)}!"
        )
        val_loss, val_cls_metrics, val_ogb_eval_res, val_triplet = evaluate(
            model,
            problem_type,
            device,
            num_labels,
            valid_loader,
            task_level,
            tokenizer_config["dataset"],
            tensor_shape_list=valid_tensor_shape_list,
        )
        print(
            f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Evaluate on full test data {test_cnt} -> {len(test_sampler)}!"
        )
        test_loss, test_cls_metrics, test_ogb_eval_res, test_triplet = evaluate(
            model,
            problem_type,
            device,
            num_labels,
            test_loader,
            task_level,
            tokenizer_config["dataset"],
            tensor_shape_list=test_tensor_shape_list,
        )
        print(
            f"[{datetime.now()}][epoch {epoch}][local {i}][global {j}] train_loss: {tr_loss}, valid_loss: {val_loss}, test_loss: {test_loss}, {test_cls_metrics.results_in_details()},\ntrain ogb_eval: {tr_ogb_eval_res}, valid ogb_eval: {val_ogb_eval_res}, test ogb_eval: {test_ogb_eval_res}"
        )
        ls_log.append(
            f"{epoch},{i},{j},{tr_loss},{val_loss},{test_loss},{','.join(val_cls_metrics.results_in_str_tuple())},{format_ogb_output_for_csv(val_ogb_eval_res)},{','.join(test_cls_metrics.results_in_str_tuple())},{format_ogb_output_for_csv(test_ogb_eval_res)}\n"
        )
        curr_lr = lr_scheduler.get_lr() if lr_scheduler is not None else [lr]
        ls_result.append(
            f"{epoch},{j},{curr_lr},{curr_lr[0]},{format_ogb_output_for_csv(tr_ogb_eval_res)},{format_ogb_output_for_csv(val_ogb_eval_res)},{format_ogb_output_for_csv(test_ogb_eval_res)}\n"
        )
        misc_utils.save_ckp(output_dir, model, epoch, use_ddp) if not bool(
            eval_only
        ) else None
        if int(os.environ.get("RANK", 0)) == 0:
            misc_utils.save_all(
                output_dir,
                model,
                epoch,
                save_model=False,
                ls_log=ls_log,
                ls_result=ls_result,
                ls_loss=ls_loss,
                tr_dict=tr_triplet if save_pred else None,
                val_dict=val_triplet if save_pred else None,
                test_dict=test_triplet if save_pred else None,
            )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    fire.Fire(train)
