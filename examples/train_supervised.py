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
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter

import sys

# sys.path.insert(0, "..")
sys.path.insert(0, ".")  # for submit to PAI
# put below `src.data` import above other `src.` to avoid `common_io` import error
from src.data import collator, vocab_builder, tokenizer, read_dataset
from src.models import (
    GraphGPTConfig,
    GraphGPTDoubleHeadsModel,
    GraphGPT2Config,
    GraphGPT2DoubleHeadsModel,
)
from src.utils import (
    conf_utils,
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
    ogb_utils,
    evaluate_ogb,
    format_ogb_output_for_csv,
    set_up_shuffle_and_sampler,
    worker_init_fn_seed,
)


dict_models = {
    "graphgpt2": (GraphGPT2DoubleHeadsModel, GraphGPT2Config),
    "graphgpt": (GraphGPTDoubleHeadsModel, GraphGPTConfig),
}


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
            task_labels=task_labels,
        )  # Perform a single forward pass.
        # record loss
        test_loss += res.task_loss
        # record metrics
        idx = test_data["idx"].to(device)
        cls_metrics.update(res.task_logits, task_labels, idx)
    print(f"[eval mode] input_ids shape: {input_ids.shape}")
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

    if dataset_name in ogb_utils._eval._register_map:
        ogb_eval_res = evaluate_ogb(dataset_name, ogb_input_dict)
    else:
        ogb_eval_res = cls_metrics.results_in_dict()
    return test_loss, cls_metrics, ogb_eval_res, ogb_input_dict


def train(
    output_dir: str = "../exp/models/graph_llama_classification_test",
    pretrain_cpt: str = "",
    data_dir: str = "../data/TUDataset",
    dataset_name: str = "reddit_threads",
    save_pred: int = 0,  # whether to save prediction results
    # tokenization config
    tokenizer_class: str = "GSTTokenizer",  # GSTTokenizer|StackedGSTTokenizer
    tokenization_config: str = "reddit_tokenization_config.json",
    attr_assignment: str = "",
    attr_shuffle: int = 1,
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
    epoch_per_eval: int = 1,
    k_samplers: int = 262144,  # 2^14=16384  2^16=65536  2^18=262144
    true_valid: int = 0,
    # deepspeed config
    deepspeed_config: str = "",
    gradient_accumulation_steps: int = 1,
    local_rank: int = 0,
    # architecture cofig
    model_type: str = "graphgpt",  # graphgpt|graphgpt2|graphbert
    model_config: str = "",
    vocab_size: int = None,  # defaults to 32000
    hidden_size: int = 128,  # defaults to 4096
    num_hidden_layers: int = 2,  # defaults to 32
    intermediate_size: int = 0,  # defaults to 11008
    num_attention_heads: int = 0,  # defaults to 32
    hidden_act: str = "silu",  # defaults to "silu"
    stacked_feat: int = 1,
    stacked_feat_agg_method: str = "sum",
    max_position_embeddings: int = 128,  # defaults to 2048
    initializer_range: float = 0.02,  # defaults to 0.02
    causal_attention: int = 1,  # use causal or bi attention
    attention_dropout: float = 0,  # used for transformers v>=4.38
    max_length: int = 1024,
    # supervised task config
    num_labels: Optional[int] = 2,
    mlp: Optional[List[int]] = None,
    pooling_method: str = "last",
    dropout: float = 0,  # dropout for mlp layers
    problem_type: Optional[
        str
    ] = None,  # single_label_classification|multi_label_classification|regression
    loss_type: Optional[str] = "",  # auc
    task_level: Optional[str] = "graph",  # pretrain|graph|edge|node
    task_ratio: float = 1,  # multi-task loss setting, ratio of sv task
    # odps config
    tables: str = "",  # ODPS input table names
    outputs: str = "",  # ODPS output table names
    # others
    samples_per_eval: int = 0,
    seed: int = 0,
):
    if (intermediate_size == 0) and (num_attention_heads == 0):
        (
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_hidden_layers,
        ) = modules_utils.set_up_model_architect(
            hidden_size=hidden_size, num_hidden_layers=num_hidden_layers
        )
    ntp_ratio, task_ratio = 1 - task_ratio, task_ratio
    assert len(deepspeed_config) > 0
    min_lr = 0 * lr
    gpu_name = torch.cuda.get_device_name()

    GraphModel, GraphModelConfig = dict_models[model_type]
    raw_pretrain_cpt = pretrain_cpt
    if os.path.exists(os.path.join(output_dir, "log.csv")):
        print(
            f"log file {os.path.join(output_dir, 'log.csv')} exists, resume training from {output_dir} instead of initializing from pre-train ckp {pretrain_cpt}!"
        )
        pretrain_cpt = output_dir
    # 0. init distributed train and get gpu/device info
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = os.environ.get("LOCAL_RANK")
    print(f"\nworld size: {world_size}, rank: {rank}, local rank: {local_rank}")
    rnd_seed = torch.random.initial_seed() - rank
    random.seed(rnd_seed)
    print(f"seed `random` with {rnd_seed}")
    eval_steps = samples_per_eval // (world_size * batch_size)
    params = print_params(**locals())

    # 1. prepare data & tokenizer
    # 1.1 set-up tokenization config
    task_type = task_level
    tokenizer_config = conf_utils.parse_tokenization_config_for_ft(
        tokenization_config=tokenization_config,
        pretrain_cpt=pretrain_cpt,
        data_dir=data_dir,
        dataset_name=dataset_name,
        task_type=task_type,
        tokenizer_class=tokenizer_class,
        attr_assignment=attr_assignment,
        attr_shuffle=attr_shuffle,
    )
    assert (
            tokenizer_config["semantics"]["attr_assignment"]
            in tokenizer_utils.ATTR_ASSIGNMENT_TYPES
    )
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
        with_prob=False,
        # for odps data reading
        table=tables,
        mode="all",
        supervised_task=task_level,
        true_valid=true_valid,
    )
    for dataset in [train_dataset, valid_dataset, test_dataset]:
        if isinstance(dataset, IterableDataset):
            print(next(iter(dataset)))
        else:
            idx = dataset.sampler[0]
            print(dataset[idx])
    # 1.3 build vocab and then init tokenizer from the tokenization config
    vocab_builder.build_vocab(raw_dataset, tokenizer_config, rank)
    tokenizer_cls = getattr(tokenizer, tokenizer_config["tokenizer_class"])
    gtokenizer = tokenizer_cls(tokenizer_config)
    inspect_tokenization_results(train_dataset, gtokenizer)
    # 1.4 get train/test sampler
    (
        train_cnt,
        train_sampler,
        train_sampler_for_eval,
        train_shuffle,
        valid_cnt,
        valid_sampler,
        valid_sampler_for_eval,
        valid_tensor_shape_list,
        test_cnt,
        test_sampler,
        test_sampler_for_eval,
        test_tensor_shape_list,
        steps_per_epoch,
    ) = loader_utils.get_train_valid_test_sampler(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        k_samplers=k_samplers,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
    )
    print(f"steps_per_epoch: {steps_per_epoch}")
    # due to `drop_last` in train_loader, use //
    samples_per_gpu = (len(train_sampler) if train_sampler else train_cnt) // world_size
    total_num_steps = epochs * (samples_per_gpu // batch_size)
    warmup_num_steps = int(warmup_epochs * (samples_per_gpu // batch_size))
    print(
        f"\ntotal_num_steps: {total_num_steps}\nwarmup_num_steps: {warmup_num_steps}\n"
    )

    # 2. set model
    # 2.1 init model config
    config = conf_utils.parse_model_config_for_ft(loss_utils=loss_utils, **locals())
    print(f"\nFinal model config for supervised task:\n{pformat(config)}\n")
    # 2.2 create model
    deepspeed.init_distributed(dist_backend="nccl", rank=rank, world_size=world_size)
    model = GraphModel(config)
    model.gradient_checkpointing_enable()
    # enable gradient_checkpointing for Llama backbone
    # silence the warnings. Please re-enable for inference!
    model.config.use_cache = False
    if freeze > -1:  # 0->freeze embedding; 1->embed+1st layer
        modules_utils.freeze_llama_layers(model, freeze)
    print_trainable_parameters(model)
    # 2.21 load from ckp IF provided existing ckp and NOT resume from the ckp
    model = loader_utils.load_from_ckp(
        misc_utils=misc_utils,
        pretrain_cpt=pretrain_cpt,
        output_dir=output_dir,
        model=model,
        config=config,
    )
    print(model)
    # 2.3 Create optimizer (load optimization config if given)
    (
        ds_config,
        non_ds_scheduler,
        scheduler_conf,
    ) = conf_utils.parse_deepspeed_config_for_ft(loss_utils=loss_utils, **locals())
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        lr_scheduler=non_ds_scheduler,
        config=ds_config,
        mpu=None,
        dist_init_required=False,
    )
    device = model.device
    # 2.4 Load model parameters and optimizer stats from ckp IF resuming from current ckp
    if (len(pretrain_cpt) > 0) and (pretrain_cpt == output_dir):
        ckp, prev_epoch = misc_utils.get_latest_ckp(pretrain_cpt)
        print(f"Loading pretrained weights from ckp {ckp}")
        model.load_checkpoint(ckp)
        print(
            f"After loading weights from ckp:\n{model.__dict__['module'].config}\nnum_labels: {model.num_labels}\nmodel-type: {model.dtype}\n\n{model}"
        )
    if int(os.environ.get("RANK", 0)) == 0:
        model.__dict__["module"].config.save_pretrained(output_dir)
    # 3. set initial status
    # 3.0 set initial condition of optimization, either resuming from ckp or starting from scratch
    (
        last_step_index,
        ep_init,
        j_init,
        ls_log,
        ls_result,
        ls_loss,
    ) = conf_utils.init_log_conf_for_ft(
        misc_utils=misc_utils,
        pretrain_cpt=pretrain_cpt,
        output_dir=output_dir,
        steps_per_epoch=steps_per_epoch,
    )

    # 3.1 init collator
    collator_fn = collator.DataCollatorForGSTCausal(
        tokenizer=gtokenizer,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
        is_training=False,
    )
    batch_size_eval = 16  # small to avoid OOM when evaluating
    num_workers_eval = min(num_workers, 16)
    train_loader_for_eval, valid_loader, test_loader = loader_utils.get_eval_loader(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        batch_size_eval=batch_size_eval,
        train_sampler_for_eval=train_sampler_for_eval,
        valid_sampler=valid_sampler,
        test_sampler=test_sampler,
        num_workers_eval=num_workers_eval,
        collator_fn=collator_fn,
    )

    tb_writer = None
    if int(os.environ.get("RANK", 0)) == 0:
        tmp_ds_config = copy.deepcopy(model.config)
        tmp_ds_config.update(scheduler_conf)
        conf_utils.dump_all_conf(**locals())

        # note: ONLY worker 0 write summary
        # flush_secs: automatic flush, default 120s
        # max_queue: queue size for storing events, default 10; >10 will flush data once to filesystem
        summary_dir = os.environ.get("SUMMARY_DIR", os.path.join(output_dir, "summary"))
        tb_writer = SummaryWriter(log_dir=summary_dir, max_queue=30, flush_secs=120)
        print(f"start logging in dir: {summary_dir}")

    # 4. Training & Inferring
    i = 0
    j = j_init
    print(
        f"[{datetime.now()}] Training start with j_init {j_init} and ep_init {ep_init} ..."
    )
    if True:
        print(f"running eval before training starts ...")
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
            f"tr_loss: {val_loss}\ntr_cls_metrics: {val_cls_metrics.results_in_details()}\ntr_ogb_eval_res: {val_ogb_eval_res}, tr_triplet: {val_triplet}"
        )
    for epoch in range(ep_init, epochs):
        if not eval_only:
            model.train()
            print(
                f"Re-initialize train-loader with shuffled sampler and reset dataset!"
            )
            if not isinstance(train_dataset, IterableDataset):
                train_dataset.reset_samples(epoch) if hasattr(
                    train_dataset, "reset_samples"
                ) else None
                train_sampler = loader_utils.distribute_sampler_with_rnd_seed(
                    train_dataset.sample_idx, world_size, rank, seed=seed + epoch
                )
            collator_fn.is_training = True
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
                    f"[sample idx top 10][local i:{i}]{data['idx'][:10]} {data['input_ids'].shape}"
                ) if i == 0 else None
                input_ids = data["input_ids"].to(device)
                attention_mask = data["attention_mask"].to(device)
                labels = data["labels"].to(device)
                task_labels = data[f"{task_level}_labels"].to(device)
                task_labels = (
                    task_labels.float()
                    if problem_type == "multi_label_classification"
                    else task_labels
                )

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pretrain_labels=labels if ntp_ratio > 0 else None,
                    task_labels=task_labels,
                )  # Perform a single forward pass.
                ntp_loss = output.pretrain_loss
                task_loss = output.task_loss
                if ntp_ratio > 0:
                    loss = ntp_loss.float() * ntp_ratio + task_loss.float() * task_ratio
                else:
                    loss = task_loss.float()
                model.backward(loss)  # Derive gradients.
                model.step()
                if j % logging_steps == 0:
                    t_interval = (datetime.now() - t_start).total_seconds()
                    samples_per_second = round(i * batch_size / t_interval)
                    print(
                        f"[{datetime.now()}][epoch {epoch}][local {i}][global {j}] processed {samples_per_second} samples per second!"
                    )
                    print(
                        f"{' ' * 8}loss: {round(loss.item(), 7)}, ntp_Loss {ntp_loss}, task_Loss {round(task_loss.item(), 7)}"
                    )
                    # Reduce SUM to get the loss from all the GPUs to RANK=0
                    # refer: https://github.com/microsoft/DeepSpeed/discussions/2377#discussioncomment-3765282
                    dist.reduce(loss, 0)
                    loss = loss / world_size
                    ls_loss.append(f"{epoch},{i},{j},{ntp_loss},{task_loss},{loss}\n")

                    tb_writer.add_scalar(
                        "loss", loss.item(), j
                    ) if tb_writer is not None else None
                j += 1
        else:
            ckp = os.path.join(pretrain_cpt, f"epoch_{epoch}")
            if os.path.exists(ckp):
                print(f"Loading weights from ckp {ckp} for evaluation")
                model = model.from_pretrained(ckp, config=config).to(device)
            else:
                print(f"ckp {ckp} doesn't exists, skip it!")
        if (epoch + 1) % epoch_per_eval == 0:
            print(
                f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Evaluate on partial train data {k_samplers * world_size} -> {k_samplers}!"
            )
            tr_loss, tr_cls_metrics, tr_ogb_eval_res, tr_triplet = (
                evaluate(
                    model,
                    problem_type,
                    device,
                    num_labels,
                    train_loader_for_eval,
                    task_level,
                    tokenizer_config["dataset"],
                    tensor_shape_list=[k_samplers] * world_size,
                )
                if train_loader_for_eval
                else (0, None, None, None)
            )
            print(
                f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Evaluate on full valid data {valid_cnt} -> {len(valid_sampler) if valid_sampler else valid_cnt / world_size}!"
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
                f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Evaluate on full test data {test_cnt} -> {len(test_sampler) if test_sampler else test_cnt / world_size}!"
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
            misc_utils.save_ckp(output_dir, model, epoch, True) if not bool(
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
            if tb_writer is not None:
                # Log histograms of model parameters
                for name, param in model.named_parameters():
                    tb_writer.add_histogram(name, param, epoch)
    tb_writer.close() if tb_writer is not None else None


if __name__ == "__main__":
    # https://github.com/pytorch/pytorch/issues/3492
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    fire.Fire(train)
