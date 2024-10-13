import os
import copy
import random
import torch
import fire
import multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
import deepspeed
from datetime import datetime
from typing import Optional, List
from pprint import pprint, pformat
from torch.utils.data import DataLoader, IterableDataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter
from timm.models import load_checkpoint
from timm.utils.model import unwrap_model, get_state_dict

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
    patch_utils,
    conf_utils,
    tokenizer_utils,
    loader_utils,
    modules_utils,
    misc_utils,
    loss_utils,
    metrics_utils,
    print_trainable_parameters,
    print_params,
    inspect_tokenization_results,
    get_metrics,
    ogb_utils,
    evaluate_ogb,
    format_ogb_output_for_csv,
    worker_init_fn_seed,
)

ModelEmaV3 = patch_utils.ModelEmaV3

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
    metric_type="",
    ds_split="train",
):
    model.eval()
    if not metric_type:
        metric_type = problem_type
    cls_metrics = get_metrics(metric_type, device, num_labels=num_labels)
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
        cls_idx = test_data["cls_idx"].to(device) if "cls_idx" in test_data else None
        inputs_raw_embeds = None
        if "embed" in test_data:
            inputs_raw_embeds = test_data["embed"].to(device)
        res = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_labels=task_labels,
            cls_idx=cls_idx,
            inputs_raw_embeds=inputs_raw_embeds,
        )  # Perform a single forward pass.
        # record loss
        test_loss += res.task_loss
        # record metrics
        # `idx` is finalized inside `collator`
        idx = test_data["idx"].to(device)
        if "raw_node_idx" in test_data:
            idx = (idx, test_data["raw_node_idx"].to(device))
        cls_metrics.update(res.task_logits, task_labels, idx)
    print(f"[{datetime.now()}][FINISHED][EVAL] input_ids shape: {input_ids.shape}")
    test_loss = test_loss / j
    cls_metrics.compute()
    ogb_input_dict = cls_metrics.to_dict()
    if dist.get_world_size() > 1:
        print(f"[{datetime.now()}][GATHERING] start gathering result tensors ...")
        # gather results from different gpus if there are more than 1 gpu
        # refer to: https://stackoverflow.com/a/71433508/4437068
        for key, val in ogb_input_dict.items():
            ogb_input_dict[key] = tensor_out = misc_utils.all_gather(val.to(device))
            cnt_nans = torch.isnan(tensor_out).sum().item()
            print(
                f"[{datetime.now()}] Update ogb_input_dict element {key} with val of shape {tensor_out.shape} of type {tensor_out.dtype} of NANs {cnt_nans}"
            )

    if dataset_name in ogb_utils._eval._register_map:
        if (ds_split == "train") and (
            dataset_name in {"ogbl-citation2", "ogbl-wikikg2"}
        ):
            # ONLY these ds' valid/test shall be evaluated
            ogb_eval_res = None
        else:
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
    tokenizer_class: str = None,  # GSTTokenizer|StackedGSTTokenizer
    tokenization_config: str = "reddit_tokenization_config.json",
    stack_method: str = None,
    attr_assignment: str = "first",
    attr_shuffle: int = 0,
    # training config
    optimization_config: str = "",
    epochs: int = 1,
    warmup_epochs: float = 0.25,
    batch_size: int = 128,
    batch_size_eval: int = 16,  # small to avoid OOM when evaluating
    pad_to_multiple_of: int = 8,
    lr: float = 0.0001,
    min_lr: float = 0,
    eps: float = 1e-8,
    betas: List[float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    logging_steps: int = 100,
    num_workers: int = 8,  # num of workers for data processing in DataLoader
    freeze: int = -1,  # how to freeze the params of backbone architecture: -1->no, 0->embedding
    # eval config
    infer_only: int = 0,
    eval_only: int = 0,
    epoch_per_eval: int = 1,
    k_samplers: int = 262144,  # 2^14=16384  2^16=65536  2^18=262144
    true_valid: int = -1,
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
    stacked_feat_agg_method: str = "gated",  # sum|gated
    max_position_embeddings: int = 128,  # defaults to 2048
    initializer_range: float = 0.02,  # defaults to 0.02
    causal_attention: int = 1,  # use causal or bi attention
    attention_dropout: float = 0,  # used for transformers v>=4.38
    embed_dropout: float = 0,
    path_dropout: float = 0,
    mlp_dropout: float = 0,
    layer_scale_init_value: float = 0,
    use_ema: int = 0,
    ema_decay: float = 0.9999,
    max_length: int = 1024,
    # supervised task config
    num_labels: Optional[int] = 2,
    mlp: Optional[List[int]] = tuple(),
    pooling_method: str = "last",
    dropout: float = 0,  # dropout for mlp layers
    problem_type: Optional[
        str
    ] = None,  # single_label_classification|multi_label_classification|regression
    metric_type: str = "",
    loss_type: Optional[str] = "",  # auc
    task_level: Optional[str] = "graph",  # pretrain|graph|edge|node
    task_ratio: float = 1,  # multi-task loss setting, ratio of sv task
    # odps config
    tables: str = "",  # ODPS input table names
    outputs: str = "",  # ODPS output table names
    # others
    samples_per_eval: int = 0,
    seed: Optional[int] = None,
):
    if infer_only:
        eval_only = 1
    if seed is None:
        seed = int(datetime.now().date().strftime("%Y%m%d")[::-1])
    use_tb_writer = False
    use_ema = bool(use_ema)
    ema_file = "model_ema.pt"
    ema_file_best = "model_ema_best.pt"
    ema_best_res = None
    ema_best_flag = False
    use_deepspeed = len(deepspeed_config) > 0

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
    if tokenizer_config["tokenizer_class"] == "StackedGSTTokenizer":
        attr_dim = (
            tokenizer_config["semantics"]["edge"]["dim"]
            + tokenizer_config["semantics"]["node"]["dim"]
        )
        assert stack_method in ("short", "long", None), f"stack_method: {stack_method}"
        if tokenizer_config["structure"]["edge"]["remove_edge_type_token"]:
            stacked_feat = 1 + attr_dim
        else:
            stacked_feat = 2 + attr_dim
    else:
        stacked_feat = 1
    embed_dim = tokenizer_config["semantics"]["node"].get(
        "embed_dim", 0
    ) + tokenizer_config["semantics"]["edge"].get("embed_dim", 0)
    print(f"stacked_feat: {stacked_feat}, embed_dim: {embed_dim}")
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
        edge_dim=tokenizer_config["semantics"]["edge"]["dim"],
        node_dim=tokenizer_config["semantics"]["node"]["dim"],
        # others
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
    gtokenizer = tokenizer_cls(
        tokenizer_config,
        stack_method=stack_method,
        loss_type=loss_type,
        num_labels=num_labels,
    )  # loss_type & num_labels -> kwargs
    inspect_tokenization_results(train_dataset, gtokenizer)
    # 1.4 get train/valid/test sampler
    (
        train_cnt,
        train_sampler,
        train_sampler_for_eval,
        train_shuffle,
        valid_cnt,
        valid_sampler,
        valid_sampler_for_eval,
        test_cnt,
        test_sampler,
        test_sampler_for_eval,
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
    if use_deepspeed:
        deepspeed.init_distributed(
            dist_backend="nccl", rank=rank, world_size=world_size
        )
    model = GraphModel(config)
    model.gradient_checkpointing_enable()
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
    model_ema = None
    if use_ema:
        model_ema = ModelEmaV3(model, decay=ema_decay)
    # 2.3 Create optimizer (load optimization config if given)
    # obtain layerwise lr
    model_parameters = model.parameters()
    # model_parameters = loss_utils.get_layerwise_param_groups(model, lr, 0.95)
    if use_deepspeed:
        (
            ds_config,
            non_ds_scheduler,
            scheduler_conf,
        ) = conf_utils.parse_deepspeed_config_for_ft(loss_utils=loss_utils, **locals())
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            lr_scheduler=non_ds_scheduler,
            config=ds_config,
            mpu=None,
            dist_init_required=False,
        )
        device = model.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DDP(model.to(device), find_unused_parameters=False)
        optimizer = torch.optim.AdamW(
            model_parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        lr_scheduler_generator, _ = loss_utils.set_py_scheduler(
            "OneCycleLR",
            {"scheduler": {"params": {}}},
            max_lr=lr,
            min_lr=min_lr,
            total_steps=total_num_steps,
            pct_start=warmup_num_steps / total_num_steps,
            last_step_index=-1,
        )
        lr_scheduler = lr_scheduler_generator(optimizer)
        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()
    if use_ema:
        model_ema.module.to(device=device)
        print(
            f"[Debug] model-ema embedding_params:\n{model_ema.module.model.embed_tokens.weight.data}"
        )
    # 2.4 Load model parameters and optimizer stats from ckp IF resuming from current ckp
    if (len(pretrain_cpt) > 0) and (pretrain_cpt == output_dir) and (not eval_only):
        ckp, prev_epoch = misc_utils.get_latest_ckp(pretrain_cpt)
        print(f"Loading pretrained weights from ckp {ckp}")
        if use_deepspeed:
            model.load_checkpoint(ckp)
        else:
            misc_utils.load_ddp_ckp(
                ckp, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
            )
        print(
            f"After loading weights from ckp:\n{model.module.config}\nnum_labels: {model.module.num_labels}\nmodel-type: {model.module.dtype}\n\n{model.module}"
        )
        if model_ema is not None:
            ema_ckp = os.path.join(output_dir, ema_file)
            load_checkpoint(model_ema.module, ema_ckp, use_ema=True)
            print(f"load model_ema ckp from {ema_ckp}")
    if (rank == 0) and (not eval_only):
        model.module.config.save_pretrained(output_dir)
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
        eval_only=eval_only,
    )

    # 3.1 init collator
    collator_fn = collator.DataCollatorForGSTCausal(
        tokenizer=gtokenizer,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
        is_training=False,
    )
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
    if (rank == 0) and (not eval_only):
        tmp_ds_config = None
        if use_deepspeed:
            tmp_ds_config = copy.deepcopy(model.config)
            tmp_ds_config.update(scheduler_conf)
        conf_utils.dump_all_conf(**locals())

        if use_tb_writer:
            # note: ONLY worker 0 write summary
            # flush_secs: automatic flush, default 120s
            # max_queue: queue size for storing events, default 10; >10 will flush data once to filesystem
            summary_dir = os.environ.get(
                "SUMMARY_DIR", os.path.join(output_dir, "summary")
            )
            tb_writer = SummaryWriter(log_dir=summary_dir, max_queue=30, flush_secs=120)
            print(f"start logging in dir: {summary_dir}")

    # 4. Training & Inferring
    i = 0
    j = j_init
    print(
        f"[{datetime.now()}] Training start with j_init {j_init} and ep_init {ep_init} ..."
    )
    if not eval_only:
        print(f"[{datetime.now()}] running eval before training starts ...")
        val_loss, val_cls_metrics, val_ogb_eval_res, val_triplet = evaluate(
            model,
            problem_type,
            device,
            num_labels,
            valid_loader,
            task_level,
            tokenizer_config["dataset"],
            metric_type=metric_type,
            ds_split="valid",
        )
        print(
            f"[{datetime.now()}] tr_loss: {val_loss}\ntr_cls_metrics: {val_cls_metrics.results_in_details()}\ntr_ogb_eval_res: {val_ogb_eval_res}, tr_triplet: {val_triplet}"
        )
        if rank == 0:
            misc_utils.save_all(
                output_dir,
                model,
                epoch=-1,
                save_model=False,
                ls_log=None,
                ls_result=None,
                ls_loss=None,
                tr_dict=None,
                val_dict=val_triplet if save_pred else None,
                test_dict=None,
            )
        ema_best_res = val_ogb_eval_res
    if eval_only:
        ep_init = min(ep_init, epochs - 1)
        print(
            f"[{datetime.now()}] In eval only mode, ep_init: {ep_init}, epochs: {epochs}!"
        )
    for epoch in range(ep_init, epochs):
        if not eval_only:
            model.train()
            print(
                f"Re-initialize train-loader with shuffled sampler and reset dataset!"
            )
            if not isinstance(train_dataset, IterableDataset):
                train_dataset.reset_samples(epoch, seed) if hasattr(
                    train_dataset, "reset_samples"
                ) else None
                train_sampler = loader_utils.distribute_sampler_with_rnd_seed(
                    torch.tensor(train_dataset.sampler),
                    world_size,
                    rank,
                    seed=seed + epoch,
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
                    f"[sample idx top 10][local i:{i}]{data['idx'][:10]} {data['input_ids'].shape}\n"
                    f"inputs keys: {data.keys()}"
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
                cls_idx = data["cls_idx"].to(device) if "cls_idx" in data else None
                inputs_raw_embeds = None
                if embed_dim > 0:
                    inputs_raw_embeds = data["embed"].to(device)
                sample_wgt = None
                if "wgt" in data:
                    sample_wgt = data["wgt"].to(device)

                if use_deepspeed:
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pretrain_labels=labels if ntp_ratio > 0 else None,
                        task_labels=task_labels,
                        cls_idx=cls_idx,
                        inputs_raw_embeds=inputs_raw_embeds,
                        sample_wgt=sample_wgt,
                    )  # Perform a single forward pass.
                    ntp_loss = output.pretrain_loss
                    task_loss = output.task_loss
                    if ntp_ratio > 0:
                        loss = (
                            ntp_loss.float() * ntp_ratio
                            + task_loss.float() * task_ratio
                        )
                    else:
                        loss = task_loss.float()
                    model.backward(loss)  # Derive gradients.
                    model.step()
                else:
                    assert (
                        gradient_accumulation_steps == 1
                    ), "https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation"
                    optimizer.zero_grad()  # Clear gradients.
                    # https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
                    # Enables autocasting for the forward pass (model + loss)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pretrain_labels=labels if ntp_ratio > 0 else None,
                            task_labels=task_labels,
                            cls_idx=cls_idx,
                            inputs_raw_embeds=inputs_raw_embeds,
                            sample_wgt=sample_wgt,
                        )  # Perform a single forward pass.
                        ntp_loss = output.pretrain_loss
                        task_loss = output.task_loss
                        if ntp_ratio > 0:
                            loss = ntp_loss * ntp_ratio + task_loss * task_ratio
                        else:
                            loss = task_loss
                    # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                    # Backward passes under autocast are not recommended.
                    # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                    scaler.scale(loss).backward()
                    if max_grad_norm > 0:
                        # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                        # Unscales the gradients of optimizer's assigned params in-place
                        scaler.unscale_(optimizer)
                        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )

                    # IF not unscaled, scaler.step() first unscales the gradients of the optimizer's assigned params.
                    # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    scaler.step(optimizer)

                    # Updates the scale for next iteration.
                    scaler.update()
                    lr_scheduler.step()
                if model_ema is not None:
                    model_ema.update(model.module, step=j)
                if j % logging_steps == 0:
                    t_interval = (datetime.now() - t_start).total_seconds()
                    samples_per_second = round(i * batch_size / t_interval)
                    print(
                        f"[{datetime.now()}][epoch {epoch}][local {i}][global {j}] processed {samples_per_second} samples per second!\n"
                        f"{' ' * 8}loss: {round(loss.item(), 7)}, ntp_Loss {ntp_loss}, task_Loss {round(task_loss.item(), 7)}\n"
                        f"{' ' * 8}{lr_scheduler.get_last_lr() if hasattr(lr_scheduler, 'get_last_lr') else 'lr ...'}"
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
                loader_utils.load_from_ckp_with_try(
                    model.module, ckp, config, skip_keys=False
                )
            else:
                print(f"ckp {ckp} doesn't exists, skip it!")
        if (epoch + 1) % epoch_per_eval == 0 and (not infer_only):
            print(
                f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Evaluate on partial train data "
                f"{k_samplers*world_size} -> {k_samplers}!"
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
                    metric_type=metric_type,
                    ds_split="train",
                )
                if train_loader_for_eval
                else (0, None, None, None)
            )
            print(
                f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Evaluate on full valid data "
                f"{valid_cnt} -> {len(valid_sampler) if valid_sampler else valid_cnt // world_size}!"
            )
            val_loss, val_cls_metrics, val_ogb_eval_res, val_triplet = evaluate(
                model,
                problem_type,
                device,
                num_labels,
                valid_loader,
                task_level,
                tokenizer_config["dataset"],
                metric_type=metric_type,
                ds_split="valid",
            )
            val_ogb_eval_res_ema = None
            if model_ema is not None:
                print(
                    f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Evaluate on full valid data with "
                    f"EMA {valid_cnt} -> {len(valid_sampler) if valid_sampler else valid_cnt // world_size}!"
                )
                (
                    val_loss_ema,
                    val_cls_metrics_ema,
                    val_ogb_eval_res_ema,
                    val_triplet_ema,
                ) = evaluate(
                    model_ema,
                    problem_type,
                    device,
                    num_labels,
                    valid_loader,
                    task_level,
                    tokenizer_config["dataset"],
                    metric_type=metric_type,
                    ds_split="valid",
                )
                ema_best_flag, ema_best_res = metrics_utils.compare_metrics_res(
                    val_ogb_eval_res_ema, ema_best_res
                )
            print(
                f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Evaluate on full test data {test_cnt}"
                f" -> {len(test_sampler) if test_sampler else test_cnt // world_size}!"
            )
            model_for_test = model
            if model_ema is not None:
                model_for_test = model_ema
                print(f"Using model-ema on test data")
            test_loss, test_cls_metrics, test_ogb_eval_res, test_triplet = evaluate(
                model_for_test,
                problem_type,
                device,
                num_labels,
                test_loader,
                task_level,
                tokenizer_config["dataset"],
                metric_type=metric_type,
                ds_split="test",
            )
            print(
                f"[{datetime.now()}][epoch {epoch}][local {i}][global {j}] train_loss: {tr_loss}, "
                f"valid_loss: {val_loss}, test_loss: {test_loss}, {test_cls_metrics.results_in_details()},\n"
                f"train ogb_eval: {tr_ogb_eval_res}, valid ogb_eval: {val_ogb_eval_res}, "
                f"EMA valid ogb_eval: {val_ogb_eval_res_ema}, test ogb_eval: {test_ogb_eval_res}"
            )
            ls_log.append(
                f"{epoch},{i},{j},{tr_loss},{val_loss},{test_loss},"
                f"{','.join(val_cls_metrics.results_in_str_tuple())},"
                f"{format_ogb_output_for_csv(val_ogb_eval_res)},"
                f"{','.join(test_cls_metrics.results_in_str_tuple())},"
                f"{format_ogb_output_for_csv(test_ogb_eval_res)}\n"
            )
            curr_lr = lr_scheduler.get_lr() if lr_scheduler is not None else [lr]
            ls_result.append(
                f"{epoch},{j},{curr_lr},{curr_lr[0]},"
                f"{format_ogb_output_for_csv(tr_ogb_eval_res)},"
                f"{format_ogb_output_for_csv(val_ogb_eval_res)},"
                f"{format_ogb_output_for_csv(test_ogb_eval_res)},"
                f"{format_ogb_output_for_csv(val_ogb_eval_res_ema)}\n"
            )
            if not eval_only:
                misc_utils.save_ckp(
                    output_dir,
                    model,
                    epoch,
                    use_deepspeed,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )
            if int(os.environ.get("RANK", 0)) == 0:
                misc_utils.save_all(
                    output_dir,
                    model,
                    epoch,
                    save_model=False,
                    ls_log=ls_log if not eval_only else None,
                    ls_result=ls_result if not eval_only else None,
                    ls_loss=ls_loss if not eval_only else None,
                    tr_dict=tr_triplet if save_pred else None,
                    val_dict=val_triplet if save_pred else None,
                    test_dict=test_triplet if save_pred else None,
                )
                if model_ema is not None:
                    ema_state = get_state_dict(model_ema, unwrap_model)
                    ema_ckp = os.path.join(output_dir, ema_file)
                    torch.save(ema_state, ema_ckp)
                    if ema_best_flag:
                        torch.save(ema_state, os.path.join(output_dir, ema_file_best))
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
