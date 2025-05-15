import os
import math
import random
import torch
import fire
import copy
import multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
import deepspeed
from datetime import datetime
from typing import Optional
from pprint import pprint, pformat
from torch.utils.data import DataLoader, IterableDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter
from timm.utils import ModelEmaV3
from timm.models import load_checkpoint
from timm.utils.model import unwrap_model, get_state_dict

import sys

# sys.path.insert(0, "..")
sys.path.insert(0, ".")
# put below `src.data` import above other `src.` to avoid `common_io` import error
from src.data import (
    collator,
    vocab_builder,
    tokenizer,
    read_dataset,
    OdpsTableIterableDataset,
)
from src.models import (
    GraphGPTConfig,
    GraphGPTCausal,
    GraphGPTPosPred,
)
from src.utils import (
    metrics_utils,
    conf_utils,
    loss_utils,
    loader_utils,
    tokenizer_utils,
    modules_utils,
    misc_utils,
    print_trainable_parameters,
    print_params,
    inspect_tokenization_results,
    set_up_shuffle_and_sampler,
    worker_init_fn_seed,
)
from torchmetrics.classification import Accuracy

# `import Accuracy` must be put here to avoid common-io import error

dict_models = {
    "graphgpt": (GraphGPTCausal, GraphGPTConfig),
    "graphgpt-pos": (GraphGPTPosPred, GraphGPTConfig),
}


@torch.no_grad()
def evaluate(
    model,
    device,
    loader,
):
    try:
        world_size = dist.get_world_size()
        rank = int(dist.get_rank())
    except ValueError:
        print("In local test setting!!!\n" * 5)
        world_size = 1
        rank = 0
    model.eval()
    print("Running test data eval ...")
    ls_loss = []
    ls_aux_loss = []
    for test_data in loader:
        input_ids = test_data["input_ids"].to(device)
        attention_mask = test_data["attention_mask"].to(device)
        position_ids = test_data["position_ids"].to(device)
        labels = test_data["labels"].to(device)
        inputs_raw_embeds = None
        if "embed" in test_data:
            inputs_raw_embeds = test_data["embed"].to(device)
        res = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            inputs_raw_embeds=inputs_raw_embeds,
            position_ids=position_ids,
        )  # Perform a single forward pass.
        loss = res.head1_loss
        aux_loss = res.head2_loss
        if world_size > 1:
            dist.reduce(loss, 0)
            if aux_loss is not None:
                dist.reduce(aux_loss, 0)
            if rank == 0:
                loss = loss / world_size
                if aux_loss is not None:
                    aux_loss = aux_loss / world_size
        ls_loss.append(loss)
        if aux_loss is not None:
            if not torch.isnan(aux_loss).item():
                ls_aux_loss.append(aux_loss)

    loss = sum(ls_loss) / len(ls_loss)
    if len(ls_aux_loss) > 0:
        aux_loss = sum(ls_aux_loss) / len(ls_aux_loss)
    print(
        f"[eval mode] input_ids: {input_ids.shape}, loss: {loss}, aux_loss: {aux_loss}"
    )
    return loss, None


def get_valid_test_loader(train_dataset, collator_fn, sampler_for_eval):
    batch_size_eval = 128  # small to avoid OOM when evaluating
    num_workers_eval = 8
    loader_for_eval = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_eval,
        sampler=sampler_for_eval,
        num_workers=num_workers_eval,
        collate_fn=collator_fn,
        worker_init_fn=worker_init_fn_seed,
        pin_memory=False,
    )
    return loader_for_eval


def get_cl_sampler(sampler: list, dup: bool = False):
    if dup:
        tmp_len = len(sampler)
    else:
        tmp_len = len(sampler) // 2
    sampler = [x for ele in zip(sampler[:tmp_len], sampler[:tmp_len]) for x in ele]
    print(f"top 10 sampler for CL loss:\n{sampler[:10]}")
    return sampler


def train(
    output_dir: str = "../exp/models/graph_llama_test",
    pretrain_cpt: str = "",
    data_dir: str = "../data/TUDataset",
    dataset_name: str = "reddit_threads",
    # tokenization config
    tokenizer_class: str = None,  # GSTTokenizer|StackedGSTTokenizer|SPGSTTokenizer
    tokenization_config: str = "reddit_tokenization_config.json",
    attr_assignment: str = "first",
    add_eos: bool = False,
    task_type: str = "pretrain",
    stack_method: str = None,
    # training config
    optimization_config: str = "",
    total_tokens: float = 1e9,
    warmup_tokens: float = 1e8,
    batch_size: int = 128,
    pad_to_multiple_of: int = 8,
    pack_tokens: float = 0,
    lr: float = 0.001,
    weight_decay: float = 0.1,
    eps: float = 1e-6,
    max_grad_norm: float = 1.0,
    logging_steps: int = 100,
    num_workers: int = 8,  # num of workers for data processing in DataLoader
    # loss config
    focal_gamma: float = 0,  # 0 for CE; other vals for focal-loss
    # eval config
    do_valid: int = 0,
    # deepspeed config
    deepspeed_config: str = "",
    gradient_accumulation_steps: int = 1,
    local_rank: int = 0,
    # architecture config
    model_type: str = "graphgpt",  # graphgpt|graphgpt2|graphbert
    model_config: str = "",
    vocab_size: int = None,
    hidden_size: int = 128,
    num_hidden_layers: int = 2,
    intermediate_size: int = 0,
    num_attention_heads: int = 0,
    hidden_act: str = "gelu",  # defaults to "silu"
    stacked_feat_agg_method: str = "gated",
    max_position_embeddings: int = 128,
    initializer_range: float = 0.02,  # defaults to 0.02
    rope_theta: int = 10000,
    rope_range: int = 0,
    tie_word_embeddings: int = 0,  # defaults to False
    causal_attention: int = 0,  # 1 for causal, 0 for bi attention
    attention_dropout: float = 0,
    embed_dropout: float = 0,
    path_dropout: float = 0,
    mlp_dropout: float = 0,
    layer_scale_init_value: float = 0,
    use_ema: int = 0,
    # odps config
    tables: str = "",  # ODPS input table names
    outputs: str = "",  # ODPS output table names
    samples_per_saving: Optional[int] = None,
):
    use_tb_writer = False
    use_ema = bool(use_ema)
    ema_file = "model_ema.pt"
    ema_file_best = "model_ema_best.pt"
    ema_best_res = None
    ema_best_flag = False
    use_deepspeed = len(deepspeed_config) > 0
    if use_ema:
        do_test = 1

    if (intermediate_size == 0) and (num_attention_heads == 0):
        (
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_hidden_layers,
        ) = modules_utils.set_up_model_architect(
            hidden_size=hidden_size, num_hidden_layers=num_hidden_layers
        )
    causal_attention = 0 if task_type == "pretrain-mlm" else causal_attention
    betas = (0.9, 0.95)
    # lr * 0.1 -> from llama2 pre-train settings
    min_lr = lr * 0.1 if use_deepspeed else 0
    gpu_name = torch.cuda.get_device_name()
    GraphModel, GraphModelConfig = dict_models[model_type]
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
    print(f"seed random with {rnd_seed}")
    steps_per_saving = samples_per_saving // (world_size * batch_size)
    print(f"\nsteps_per_saving: {steps_per_saving}")
    params = print_params(**locals())

    # 1. prepare data & tokenizer
    # 1.1 read configuration
    tokenizer_config = conf_utils.parse_tokenization_config(**locals())
    assert "pretrain" in tokenizer_config["task_type"]
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
        next_n_token = stacked_feat
    else:
        stacked_feat = 1
        next_n_token = 1
    embed_dim = tokenizer_config["semantics"]["node"].get(
        "embed_dim", 0
    ) + tokenizer_config["semantics"]["edge"].get("embed_dim", 0)
    print(
        f"stacked_feat: {stacked_feat}, next_n_token: {next_n_token}, embed_dim: {embed_dim}"
    )
    smtp_inside = False
    tmp_method = tokenizer_config.get("pretrain_mlm", {}).get("method", None)
    if tmp_method == "inside_model":
        # perform SMTP mask inside the model's `forward` method
        smtp_inside = True
        assert tokenizer_config["task_type"] in {"pretrain-mlm", "pretrain-smtp"}
        tokenizer_config["task_type"] = task_type = "pretrain-smtp"
        print(f"set task_type = {task_type} for {tmp_method}")

    # 1.2 get graph dataset
    dataset, raw_dataset = read_dataset(
        name=tokenizer_config["dataset"],
        # for local data file reading
        data_dir=data_dir,
        sampling_config=tokenizer_config["sampling"],
        # for odps data reading
        table=tables,
        edge_dim=tokenizer_config["semantics"]["edge"]["dim"],
        node_dim=tokenizer_config["semantics"]["node"]["dim"],
        mode="train",
        # general
        pretrain_mode=True,
        ensemble_datasets=tokenizer_config.get("ensemble_datasets", []),
    )
    reset_samples_per_epoch = (
        dataset.reset_samples_per_epoch
        if hasattr(dataset, "reset_samples_per_epoch")
        else False
    )
    if isinstance(dataset, IterableDataset):
        print(next(iter(dataset)))
    else:
        idx = dataset.sampler[0]
        print(dataset[idx])
    # 1.3 build vocab and then init tokenizer from the tokenization config
    vocab_builder.build_vocab(raw_dataset, tokenizer_config, rank)
    tokenizer_cls = getattr(tokenizer, tokenizer_config["tokenizer_class"])
    gtokenizer = tokenizer_cls(
        tokenizer_config, add_eos=add_eos, stack_method=stack_method
    )
    # 1.4 get train/test sampler
    train_dataset = dataset
    if not isinstance(train_dataset, IterableDataset):
        train_sampler = train_dataset.sampler
        random.shuffle(train_sampler)
        train_shuffle, train_sampler, train_cnt = set_up_shuffle_and_sampler(
            train_dataset, train_sampler
        )
    else:
        train_cnt = len(train_dataset) * world_size
        train_sampler = None
        train_shuffle = False

    sampler_for_eval = []
    if do_valid:
        # v1. split here
        (
            train_sampler,
            sampler_eval,
            _,
        ) = loader_utils.obtain_deterministic_sampler_by_ratio(
            train_dataset.sample_idx, seed=42, train=0.98, valid=0.02
        )
        # v2. split from idx
        # train_sampler = raw_dataset.get_idx_split()["train"].tolist()
        # sampler_for_eval = raw_dataset.get_idx_split()["valid"].tolist()
        # sampler_for_eval = random.sample(
        #     raw_dataset.get_idx_split()["valid"].tolist(), 10000
        # )
        print(
            f"[{datetime.now()}] train samples reduced from {len(train_dataset.sampler)} -> {len(train_sampler)}"
        )
        train_dataset.sampler = train_sampler
        sampler_for_eval = loader_utils.distribute_sampler(
            sampler_eval, world_size, rank
        )
        if task_type.endswith("-cl"):
            sampler_for_eval = get_cl_sampler(sampler_for_eval, True)
        print(
            f"[{datetime.now()}] sampler_for_eval: {len(sampler_eval)}, top 10: {sampler_for_eval[:10]}"
        )

    if pack_tokens > 0:
        gtokenizer.mpe = max_position_embeddings
        # cannot pass `iter(train_dataset)` for Iterable ds, because `TypeError: cannot pickle 'generator' object`
        gtokenizer.dataset = train_dataset
        gtokenizer.sampler = tuple(train_sampler) if train_sampler is not None else None
        gtokenizer.random_ratio = pack_tokens
        tokens_per_sample = max_position_embeddings
    else:
        tokens_per_sample = misc_utils.estimate_tokens_per_sample(
            gtokenizer,
            train_dataset,
            train_sampler,
            max_position_embeddings,
            world_size,
        )
    tokens_per_sample = (
        tokens_per_sample // 2 if task_type == "pretrain-euler" else tokens_per_sample
    )
    print(f"\n[{datetime.now()}] tokens_per_sample: {tokens_per_sample}")

    inspect_tokenization_results(dataset, gtokenizer)
    # re-initialize `gtokenizer.dataset` to avoid `TypeError: cannot pickle 'generator' object`
    gtokenizer.dataset = train_dataset if pack_tokens > 0 else None

    total_num_steps = int(
        math.ceil(total_tokens / (tokens_per_sample * batch_size * world_size))
    )
    warmup_num_steps = int(
        math.ceil(warmup_tokens / (tokens_per_sample * batch_size * world_size))
    )
    tmp_cnt = len(train_sampler) if train_sampler else train_cnt / world_size
    epochs = int(math.ceil(total_tokens / (tmp_cnt * tokens_per_sample * world_size)))
    print(
        f"\n[{datetime.now()}] total_num_steps: {total_num_steps}\nwarmup_num_steps: {warmup_num_steps}\nepochs per worker: {epochs}\n"
    )

    # 2. set model
    # 2.1 init model config
    config = conf_utils.parse_model_config(**locals())
    print(config)
    # 2.2 create model
    if use_deepspeed:
        deepspeed.init_distributed(
            dist_backend="nccl", rank=rank, world_size=world_size
        )
    model = GraphModel(config)
    if hasattr(train_dataset, "dict_bounds"):  # for PCQM4M-v2 dataset ONLY
        model.dict_bounds = train_dataset.dict_bounds
    model.gradient_checkpointing_enable()
    # silence the warnings. Please re-enable for inference!
    model.config.use_cache = False
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
    model_parameters = model.parameters()
    # obtain layerwise lr
    # model_parameters = loss_utils.get_layerwise_param_groups(model, lr, 0.95)
    if use_deepspeed:
        ds_config = conf_utils.parse_deepspeed_config(loss_utils=loss_utils, **locals())
        print(f"\n[{datetime.now()}] ds_config:\n{pformat(ds_config)}")
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
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
            total_steps=total_num_steps + 1,
            pct_start=warmup_num_steps / total_num_steps,
            last_step_index=-1,
        )  # total_num_steps+1 to avoid error of lr_scheduler.step() in last step
        lr_scheduler = lr_scheduler_generator(optimizer)
        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()
    model_ema = None
    if use_ema:
        model_ema = ModelEmaV3(model.module)
    # 2.4 Load model parameters and optimizer stats from ckp IF resuming from current ckp
    if (len(pretrain_cpt) > 0) and (pretrain_cpt == output_dir):
        ckp, _ = misc_utils.get_latest_ckp(pretrain_cpt)
        print(
            f"Loading existing weights from ckp {ckp} using deepspeed API to resume training."
        )
        if use_deepspeed:
            model.load_checkpoint(ckp)
        else:
            misc_utils.load_ddp_ckp(
                ckp, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
            )
        print(f"After loading weights from ckp:\n{model.module.config}")
        if model_ema is not None:
            ema_ckp = os.path.join(output_dir, ema_file)
            load_checkpoint(model_ema.module, ema_ckp, use_ema=True)
            print(f"load model_ema ckp from {ema_ckp}")

    if int(os.environ.get("RANK", 0)) == 0:
        model.module.config.save_pretrained(output_dir)
        print(
            f"[{datetime.now()}] Finish -> Dump model config to `{output_dir}/config.json`"
        )
    print(f"[{datetime.now()}] Finish -> 2. set optimizer")
    # 3. set initial status
    # 3.0 set initial condition of optimization, either resuming from ckp or starting from scratch
    last_step_index, ep_init, j_init, ls_log, ls_result = conf_utils.init_log_conf(
        misc_utils=misc_utils,
        pretrain_cpt=pretrain_cpt,
        output_dir=output_dir,
        steps_per_saving=steps_per_saving,
    )
    # 3.1 init collator
    collator_fn = collator.DataCollatorForGSTCausal(
        tokenizer=gtokenizer,
        max_length=max_position_embeddings,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    )
    print(f"[{datetime.now()}] Finish -> 3.1 init collator")
    if do_valid:
        valid_tokenizer_config = copy.deepcopy(tokenizer_config)
        # valid_tokenizer_config["structure"]["nx"]["func"] = [
        #     {"name": "shortest_path_length", "valid": 1}
        # ]
        # gtokenizer_valid = tokenizer_cls(valid_tokenizer_config, add_eos=False)
        # if valid_tokenizer_config["structure"]["nx"]["enable"]:
        #     gtokenizer_valid.attr_mask_ratio = 1  # mask all attr
        # inspect_tokenization_results(train_dataset, gtokenizer_valid)
        # gtokenizer_valid = gtokenizer
        gtokenizer_valid = tokenizer_cls(
            valid_tokenizer_config, add_eos=add_eos, stack_method=stack_method
        )
        valid_loader = get_valid_test_loader(
            train_dataset,
            collator.DataCollatorForGSTCausal(
                tokenizer=gtokenizer_valid,
                max_length=max_position_embeddings,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ),
            sampler_for_eval,
        )
        evaluate(model, device, valid_loader)

    # 3.2 set-up loader
    tb_writer = None
    if int(os.environ.get("RANK", 0)) == 0:
        tmp_ds_config = None
        if use_deepspeed:
            tmp_ds_config = copy.deepcopy(model.config)
        conf_utils.dump_all_conf(**locals())

        if use_tb_writer:
            # note: ONLY worker 0 write summary
            # flush_secs: automatic flush, default 120s
            # max_queue: queue size for storing events, default 10; >10 will flush data once to filesystem
            # os.path.join(output_dir, "summary")   os.environ['SUMMARY_DIR']
            summary_dir = os.environ.get(
                "SUMMARY_DIR", os.path.join(output_dir, "summary")
            )
            tb_writer = SummaryWriter(log_dir=summary_dir, max_queue=30, flush_secs=120)
            print(f"start logging in dir: {summary_dir}")

    if (not reset_samples_per_epoch) and (
        not isinstance(train_dataset, IterableDataset)
    ):
        train_sampler_new = []
        for epoch in range(epochs):
            train_dataset.reset_samples(epoch, rank)
            # random.shuffle(train_sampler)
            train_sampler_new.extend(train_dataset.sampler)
        random.shuffle(train_sampler_new)
        print(
            f"train_sampler for {epochs} epochs increase: {len(train_sampler)} -> {len(train_sampler_new)}"
        )
        train_sampler = train_sampler_new
        if task_type.endswith("-cl"):
            train_sampler = get_cl_sampler(train_sampler, False)
        epochs = 1
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
        prefetch_factor=4,
    )

    # 4. Training ...
    print(f"[{datetime.now()}] Training start ...")
    is_odps_table_ds = isinstance(train_dataset, OdpsTableIterableDataset)
    if is_odps_table_ds:
        steps_per_epoch = train_cnt // (world_size * batch_size)
        print(
            f"\nsteps_per_epoch: {steps_per_epoch} = {train_cnt} // ({world_size} * {batch_size})"
        )
        epoch_start = j_init // steps_per_epoch
        print(
            f"pre-train using odps table, set epoch_start={epoch_start} in case of resuming pre-training"
        )
    else:
        epoch_start = 0
    j = j_init
    ep = ep_init
    model.train()
    for epoch in range(epoch_start, epochs):
        if (not isinstance(train_dataset, IterableDataset)) and reset_samples_per_epoch:
            print(
                f"Re-initialize train-loader with shuffled sampler and reset dataset!"
            )
            train_dataset.reset_samples(epoch, rank)
            if not do_valid:
                train_sampler = train_dataset.sampler
            random.shuffle(train_sampler)
            print(f"train_sampler: {len(train_sampler)}")
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
        i_local = 0
        if is_odps_table_ds:
            train_loader, i_local = loader_utils.init_loader_for_odps_table_ds(
                epoch=epoch,
                epoch_start=epoch_start,
                j=j,
                steps_per_epoch=steps_per_epoch,
                batch_size=batch_size,
                OdpsTableIterableDataset=OdpsTableIterableDataset,
                tables=tables,
                edge_dim=train_dataset.edge_dim,
                node_dim=train_dataset.node_dim,
                y_dim=train_dataset.y_dim,
                train_shuffle=train_shuffle,
                train_sampler=train_sampler,
                num_workers=num_workers,
                collator_fn=collator_fn,
            )
        # print(f"Top 10 samples' idx:\n{train_loader.sampler[:10]}")
        t_start = datetime.now()
        for i, data in enumerate(train_loader, i_local):
            # Iterate in batches over the training dataset
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            position_ids = data["position_ids"].to(device)
            labels = data["labels"].to(device)
            inputs_raw_embeds = None
            if embed_dim > 0:
                inputs_raw_embeds = data["embed"].to(device)

            if use_deepspeed:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    inputs_raw_embeds=inputs_raw_embeds,
                    position_ids=position_ids,
                )  # Perform a single forward pass.
                main_loss = output.head1_loss
                aux_loss = output.head2_loss
                if aux_loss is not None:
                    loss = main_loss + aux_loss
                else:
                    loss = main_loss
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
                        labels=labels,
                        inputs_raw_embeds=inputs_raw_embeds,
                        position_ids=position_ids,
                    )  # Perform a single forward pass.
                    main_loss = output.head1_loss
                    aux_loss = output.head2_loss
                    if aux_loss is not None:
                        loss = main_loss + aux_loss
                    else:
                        loss = main_loss
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward()
                if max_grad_norm > 0:
                    # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)
                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # IF not unscaled, scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()
                lr_scheduler.step()
            if model_ema is not None:
                model_ema.update(model, step=j)
            if j % logging_steps == 0:
                t_interval = (datetime.now() - t_start).total_seconds()
                samples_per_second = round((i - i_local) * batch_size / t_interval, 1)
                tokens_per_second = round(
                    (i - i_local) * batch_size * tokens_per_sample / t_interval
                )
                if aux_loss is not None:
                    loss_log = f"\n{' ' * 8}loss: {round(loss.item(), 6)}, aux_Loss {round(aux_loss.item(), 6)}, main_Loss {round(main_loss.item(), 6)}"
                else:
                    loss_log = f"loss: {round(loss.item(), 6)}"
                print(
                    f"[{datetime.now()}][epoch {ep}][local {epoch}: {i}][global {j}] train_loss: {round(loss.item(),7)}, {samples_per_second} samples / {tokens_per_second} tokens per sec"
                )
                # Reduce SUM to get the loss from all the GPUs to RANK=0
                # refer: https://github.com/microsoft/DeepSpeed/discussions/2377#discussioncomment-3765282
                dist.reduce(loss, 0)
                loss = loss / world_size
                curr_lr = lr_scheduler.get_lr() if lr_scheduler is not None else [lr]
                ls_log.append(
                    f"{ep},{curr_lr[0]},{i},{j},{aux_loss},{main_loss},{loss}\n"
                )

                tb_writer.add_scalar(
                    "loss", loss.item(), j
                ) if tb_writer is not None else None

            if ((j % steps_per_saving == 0) and (j > j_init)) or (j == total_num_steps):
                ep += 1
                print(
                    f"[{datetime.now()}][end of epoch {ep}][local {epoch}: {i}][global {j}] "
                    f"Trained with {int(j * tokens_per_sample * batch_size * world_size)} tokens! Saving ckp and logs!"
                )
                misc_utils.save_ckp(
                    output_dir,
                    model,
                    ep,
                    use_deepspeed,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )

                valid_acc = None
                val_triplet = None
                if do_valid:
                    print(f"[{datetime.now()}] Evaluating Model on valid data")
                    valid_acc, val_triplet = evaluate(model, device, valid_loader)
                    model.train()

                if int(os.environ.get("RANK", 0)) == 0:
                    # save ckp and logs
                    misc_utils.save_all(
                        output_dir,
                        model,
                        ep,
                        save_model=False,
                        ls_log=ls_log,
                        val_dict=val_triplet,
                    )
                    if model_ema is not None:
                        ema_state = get_state_dict(model_ema, unwrap_model)
                        ema_ckp = os.path.join(output_dir, ema_file)
                        torch.save(ema_state, ema_ckp)
                        if ema_best_flag:
                            torch.save(
                                ema_state, os.path.join(output_dir, ema_file_best)
                            )
                print(
                    f"[{datetime.now()}][input_id] shape: {input_ids.shape}\n"
                    f"    valid eval results: {valid_acc}\n"
                    f"    [inputs_raw_embeds(sliced)]:\n{inputs_raw_embeds[:2, :8] if inputs_raw_embeds is not None else None}"
                )
                if tb_writer is not None:
                    # Log histograms of model parameters
                    for name, param in model.named_parameters():
                        tb_writer.add_histogram(name, param, ep)

            if j == total_num_steps:
                print(
                    f"Total number of steps {total_num_steps} reached, break inner loop!!!"
                )
                break
            j += 1
        if j >= total_num_steps:
            print(
                f"Total number of steps {total_num_steps} reached, break outer loop!!!"
            )
            break

    tb_writer.close() if tb_writer is not None else None


if __name__ == "__main__":
    # https://github.com/pytorch/pytorch/issues/3492
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    fire.Fire(train)
