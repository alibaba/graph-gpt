import os

import multiprocessing as mp
import deepspeed
from datetime import datetime
from pprint import pprint, pformat
from torch.utils.data import DataLoader, IterableDataset
from deepspeed.profiling.flops_profiler import FlopsProfiler
from omegaconf import OmegaConf

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
    convert_to_legacy_config,
    GraphGPTPretrainBase,
    GraphGPTPosPred,
)
from src.utils import (
    conf_utils,
    loss_utils,
    loader_utils,
    log_eval_dump_utils,
    modules_utils,
    misc_utils,
    print_trainable_parameters,
    inspect_tokenization_results,
    worker_init_fn_seed,
    opt_utils,
    training_utils,
)
from src.utils.log_eval_dump_utils import evaluate, evaluate_generation, eval_pt_gen_only
from src.conf import (
    Config,
    OptimizingStats,
    TrainingStats,
    OdpsStats,
    EMAConfig,
    EMAStats,
    LoaderStats,
    base_configs,
)

import hydra

# `import Accuracy` must be put here to avoid common-io import error

dict_models = {
    "graphgpt": GraphGPTPretrainBase,
    "graphgpt-pos": GraphGPTPosPred,
}


# Apply Hydra decorator to parse the config file
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: Config):
    # extract main config component
    token_cfg = cfg.tokenization
    model_cfg = cfg.model
    train_cfg = cfg.training
    # extract sub-configs
    data_cfg = token_cfg.data
    sched_cfg = train_cfg.schedule
    optim_cfg = train_cfg.optimizer
    train_cfg.pretrain_mode = True
    # update cfg if needed
    train_cfg.do_valid = train_cfg.valid_percent > 0
    train_cfg.do_test = optim_cfg.use_ema
    base_configs.update_odps_cfg_from_token_cfg(cfg, "train")

    # extract params from config
    pretrain_cpt = train_cfg.pretrain_cpt
    output_dir = train_cfg.output_dir
    # tokenization config
    task_type = train_cfg.task_type
    # training config
    batch_size = train_cfg.batch_size

    use_tb_writer = False
    ema_cfg = EMAConfig(
        use_ema=optim_cfg.use_ema,
        ema_file="model_ema.pt",
        ema_file_best="model_ema_best.pt",
    )
    ema_stats = EMAStats(ema_cfg=ema_cfg)

    train_cfg.use_deepspeed = use_deepspeed = len(train_cfg.deepspeed_conf_file) > 0

    # lr * 0.1 -> from llama2 pre-train settings
    optim_cfg.min_lr = optim_cfg.lr * 0.1 if use_deepspeed else 0
    if os.path.exists(os.path.join(output_dir, "log.csv")):
        print(
            f"log file {os.path.join(output_dir, 'log.csv')} exists, resume training from {output_dir} instead of initializing from pre-train ckp {pretrain_cpt}!"
        )
        pretrain_cpt = output_dir
    # 0. set-up distributed training
    tmp_env = misc_utils.set_dist_env(cfg.training)
    world_size = train_cfg.distributed.world_size
    rank = train_cfg.distributed.rank
    sched_cfg.steps_per_saving = sched_cfg.samples_per_saving // (world_size * batch_size)

    # 1. prepare data & tokenizer
    # 1.1 read configuration
    base_configs.init_stacked_feat(cfg)
    base_configs.init_embed_dim(cfg)
    base_configs.sync_config(cfg)

    tokenizer_config = conf_utils.convert_to_legacy_tokenization_config(cfg)
    pprint(tokenizer_config)
    model_cfg.pt_head.smtp_inside = False
    tmp_method = tokenizer_config.get("pretrain_mlm", {}).get("method", None)
    if tmp_method == "inside_model":
        # perform SMTP mask inside the model's `forward` method
        model_cfg.pt_head.smtp_inside = True
        assert tokenizer_config["task_type"] in {"pretrain-mlm", "pretrain-smtp"}
        tokenizer_config["task_type"] = task_type = "pretrain-smtp"
        print(f"set task_type = {task_type} for {tmp_method}")

    # 1.2 get graph dataset
    dataset, raw_dataset = read_dataset(name=data_cfg.dataset, data_cfg=data_cfg, train_cfg=train_cfg)
    reset_samples_per_epoch = (
        dataset.reset_samples_per_epoch
        if hasattr(dataset, "reset_samples_per_epoch")
        else False
    )
    if isinstance(dataset, IterableDataset):
        tmp_sample = next(iter(dataset))
    else:
        tmp_sample = dataset[dataset.sampler[0]]
    print(f"[{datetime.now()}] Inspecting one sample:\n{tmp_sample}")

    # 1.3 get train/valid/test sampler
    train_dataset = dataset
    pt_sampler = loader_utils.get_pt_train_valid_test_sampler(
        train_dataset,
        train_cfg,
        task_type,
        data_cfg,
        read_dataset,
    )

    # 1.4 build vocab if needed
    vocab_builder.build_vocab(
        raw_dataset, tokenizer_config, cfg.training.distributed.rank
    )

    # 1.5 init tokenizer from the tokenization config
    tokenizer_cls = getattr(tokenizer, tokenizer_config["tokenizer_class"])
    gtokenizer = tokenizer_cls(
        tokenizer_config,
        add_eos=token_cfg.add_eos,
        stack_method=model_cfg.graph_input.stack_method,
        train_cfg=train_cfg,
    )
    base_configs.update_generation_cfg(cfg, gtokenizer)
    # 1.51 alter tokenizer's params based on training config
    if train_cfg.pack_tokens > 0:
        gtokenizer.mpe = model_cfg.max_position_embeddings
        # cannot pass `iter(train_dataset)` for Iterable ds, because `TypeError: cannot pickle 'generator' object`
        gtokenizer.dataset = train_dataset
        gtokenizer.sampler = (
            tuple(pt_sampler.train_sampler)
            if pt_sampler.train_sampler is not None
            else None
        )
        gtokenizer.random_ratio = train_cfg.pack_tokens
        tokens_per_sample = model_cfg.max_position_embeddings
    else:
        tokens_per_sample = misc_utils.estimate_tokens_per_sample(
            gtokenizer,
            train_dataset,
            pt_sampler.train_sampler,
            model_cfg.max_position_embeddings,
            train_cfg.distributed.world_size,
            train_cfg.tot_samples if not (train_cfg.pt_eval_only or train_cfg.do_infer) else 100,
        )
    tokens_per_sample = (
        tokens_per_sample // 2 if task_type == "pretrain-euler" else tokens_per_sample
    )
    print(f"\n[{datetime.now()}] tokens_per_sample: {tokens_per_sample}")
    # 1.52 inspect tokenization results
    inspect_tokenization_results(dataset, gtokenizer)
    # 1.53 re-initialize `gtokenizer.dataset` to avoid `TypeError: cannot pickle 'generator' object`
    gtokenizer.dataset = train_dataset if train_cfg.pack_tokens > 0 else None
    # 1.54 update training schedule config
    base_configs.update_num_steps(sched_cfg, tokens_per_sample, batch_size, world_size)
    samples_per_gpu = (
        len(pt_sampler.train_sampler)
        if pt_sampler.train_sampler
        else pt_sampler.train_cnt / world_size
    )
    base_configs.update_epochs(
        sched_cfg, tokens_per_sample, samples_per_gpu, world_size
    )
    base_configs.print_stats(sched_cfg)

    # 2. set model
    # 2.1 init model config
    model_cfg = modules_utils.set_model_config(cfg, gtokenizer)
    config = convert_to_legacy_config(model_cfg)
    print(config)
    # 2.2 create model
    if use_deepspeed:
        deepspeed.init_distributed(
            dist_backend="nccl", rank=rank, world_size=world_size
        )
    model = dict_models[model_cfg.model_type](config)
    if hasattr(train_dataset, "dict_bounds"):  # for PCQM4M-v2 dataset ONLY
        model.dict_bounds = train_dataset.dict_bounds
    model.gradient_checkpointing_enable()
    # silence the warnings. Please re-enable for inference!
    model.config.use_cache = False
    print_trainable_parameters(model)
    ## EVAL ONLY MODE
    if train_cfg.pt_eval_only:
        eval_pt_gen_only(
            model, cfg, collator.DataCollatorForGST, tokenizer_cls, tokenizer_config, pt_sampler, train_dataset
        )
        return
    ## INFER ONLY MODE
    if train_cfg.do_infer:
        # remove random mask for inference purpose
        train_cfg.pretrain_mlm.params.umr_clip = [1, 1]
        log_eval_dump_utils.pt_infer_only(
            model, cfg, collator.DataCollatorForGST, tokenizer_cls, tokenizer_config, pt_sampler
        )
        return
    # 2.21 NON-Resuming: load from ckp IF provided existing ckp other than current ckp
    model = loader_utils.load_from_ckp(
        misc_utils=misc_utils,
        pretrain_cpt=pretrain_cpt,
        output_dir=output_dir,
        model=model,
        config=config,
    )
    print(model)

    # 3. Setup training::optimizer
    model_parameters = model.parameters()
    # obtain layerwise lr
    # model_parameters = loss_utils.get_layerwise_param_groups(model, lr, 0.95)
    if use_deepspeed:
        ds_config = conf_utils.parse_deepspeed_config(
            training=train_cfg, loss_utils=loss_utils
        )
        print(f"\n[{datetime.now()}] ds_config:\n{pformat(ds_config)}")
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            config=ds_config,
            mpu=None,
            dist_init_required=False,
        )
        opt_stats = OptimizingStats(optimizer, lr_scheduler)
        prof = None
    else:
        prof = FlopsProfiler(model)
        model, opt_stats = opt_utils.initialize_optimizer(
            model=model,
            model_parameters=model_parameters,
            training=train_cfg,
            loss_utils=loss_utils,
        )
    device = model.device
    print(f"[{datetime.now()}] Finish -> 3. set optimizer")

    ema_stats.init_ema(model)

    # 4. SET initial status
    # 4.1 Resuming: load model parameters and optimizer stats from ckp from current ckp
    if (len(pretrain_cpt) > 0) and (pretrain_cpt == output_dir):
        ckp, _ = misc_utils.get_latest_ckp(pretrain_cpt)
        if use_deepspeed:
            print(f"Loading weights from {ckp} with deepspeed API to resume training.")
            model.load_checkpoint(ckp)
        else:
            misc_utils.load_ddp_ckp(
                ckp,
                model=model,
                optimizer=opt_stats.optimizer,
                lr_scheduler=opt_stats.lr_scheduler,
            )
        print(f"[{datetime.now()}] Finish -> 4.1 Loading weights from ckp:\n{model.module.config}")
        ema_stats.load_ema_ckp(output_dir)

    if int(os.environ.get("RANK", 0)) == 0:
        try:
            model.module.config.save_pretrained(output_dir)
        except AttributeError:
            print("In local test setting!!!\n" * 5)
            model.config.save_pretrained(output_dir)
        print(
            f"[{datetime.now()}] Finish -> 4.1 Dump model config to `{output_dir}/config.json`"
        )

    # 4.2 set initial condition of training, either resuming from ckp or starting from scratch
    _, ckp_init, j_init, ls_log, ls_result = conf_utils.init_log_conf(
        misc_utils=misc_utils,
        pretrain_cpt=pretrain_cpt,
        output_dir=output_dir,
        steps_per_saving=sched_cfg.steps_per_saving,
    )
    # 4.3 init collator
    collator_fn = collator.DataCollatorForGST(
        tokenizer=gtokenizer,
        max_length=model_cfg.max_position_embeddings,
        pad_to_multiple_of=train_cfg.pad_to_multiple_of,
        return_tensors="pt",
    )
    print(f"[{datetime.now()}] Finish -> 4.3 init collator")
    # 4.4 set-up loader
    # 4.41 set-up valid/test loader and evaluate before training
    valid_loader, valgen_loader = loader_utils.initialize_pt_valid_loader(
        train_dataset, cfg, pt_sampler, tokenizer_config, tokenizer_cls, collator.DataCollatorForGST
    )
    evaluate(model, valid_loader, "valid", train_cfg.do_valid)
    evaluate_generation(
        model, valgen_loader, "valid", train_cfg.do_valid and train_cfg.do_generation, cfg
    )
    test_loader = loader_utils.initialize_pt_test_loader(
        cfg, pt_sampler, tokenizer_cls, tokenizer_config, collator.DataCollatorForGST
    )
    acc, triplet = evaluate(model, test_loader, "test", train_cfg.do_test)
    ema_stats.ema_best_res = {"loss": acc} if acc else None

    # 4.42 dump ds config and initialize tensorboard in `worker 0`
    tb_writer = log_eval_dump_utils.pt_dump_cfg_and_init_tb(model, use_deepspeed, use_tb_writer, output_dir)
    # 4.43 reset train sampler if needed: a). pretraining data is very small; b). CL (Contrastive Learning)
    pt_sampler = loader_utils.reset_pt_train_sampler(
        reset_samples_per_epoch, task_type, train_dataset, train_cfg, pt_sampler
    )
    # 4.44 set-up train loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=pt_sampler.train_shuffle,
        sampler=pt_sampler.train_sampler,
        num_workers=train_cfg.num_workers,
        collate_fn=collator_fn,
        worker_init_fn=worker_init_fn_seed,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
    )

    # 5. Training ...
    print(f"[{datetime.now()}] 5. Training start ...")
    j = j_init
    ckp = ckp_init
    train_stats = TrainingStats(
        device=device,
        has_embeds_input=model_cfg.graph_input.embed_dim > 0,
        use_deepspeed=use_deepspeed,
        ls_log=ls_log,
        ls_result=ls_result,
        reset_samples_per_epoch=reset_samples_per_epoch,
        tokens_per_sample=tokens_per_sample,
        ckp=ckp,
        j=j,
        odps_stats=OdpsStats(
            is_odps_table_ds=isinstance(train_dataset, OdpsTableIterableDataset),
            tables=data_cfg.odps.tables,
        ),
    )
    loader_utils.init_odps_ds_stats(train_cfg, train_stats, pt_sampler)
    model.train()
    if not use_deepspeed:
        prof.start_profile()
    OmegaConf.save(config=cfg, f=os.path.join(output_dir, "config.yaml"))
    for epoch in range(train_stats.epoch_start, sched_cfg.epochs):
        train_stats.epoch = epoch
        train_loader = (
                loader_utils.initialize_train_loader_at_epoch_start(train_dataset, train_cfg, train_stats, pt_sampler,
                                                                    collator_fn, OdpsTableIterableDataset)
                or train_loader
        )
        loader_stats = LoaderStats(
            train_loader=train_loader,
            valid_loader=valid_loader,
            valgen_loader=valgen_loader,
            test_loader=test_loader,
        )
        # print(f"Top 10 samples' idx:\n{train_loader.sampler[:10]}")
        train_stats.t_start = datetime.now()

        with tmp_env:
            for i, data in enumerate(loader_stats.train_loader, train_stats.i_local):
                train_stats.i = i
                # Iterate in batches over the training dataset
                training_utils.batch_training(
                    data, model, train_cfg, train_stats, opt_stats
                )
                ema_stats.update_ema(model, step=train_stats.j)
                if train_stats.j % sched_cfg.logging_steps == 0:
                    log_eval_dump_utils.log_pt_training_stats(
                        train_cfg, train_stats, opt_stats, prof=prof, tb_writer=tb_writer
                    )

                if (
                    (train_stats.j % sched_cfg.steps_per_saving == 0)
                    and (train_stats.j > j_init)
                ) or (train_stats.j == sched_cfg.total_num_steps):
                    log_eval_dump_utils.log_dump_pt_training_stats(
                        model, cfg, train_stats, opt_stats, loader_stats, ema_stats, tb_writer
                    )

                if train_stats.j == sched_cfg.total_num_steps:
                    print(
                        f"Final step reached: {sched_cfg.total_num_steps}, break inner loop!"
                    )
                    break
                train_stats.j += 1
        if train_stats.j >= sched_cfg.total_num_steps:
            print(f"Final step reached: {sched_cfg.total_num_steps}, break outer loop!")
            break

    tb_writer.close() if tb_writer is not None else None
    if not use_deepspeed:
        prof.end_profile()
    OmegaConf.save(config=cfg, f=os.path.join(output_dir, "config_final.yaml"))


if __name__ == "__main__":
    # https://github.com/pytorch/pytorch/issues/3492
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    print(sys.argv)
    # `--local_rank` is injected by deepspeed, and will not be recognized by Hydra, so FILTER it
    sys.argv = [a for a in sys.argv if not a.startswith("--local_rank")]
    assert sys.argv[0].endswith(".py"), f"{sys.argv[0]}"
    if "=" not in sys.argv[-1]:
        print("Parsing space separate arguments in Nebula")
        space_args = sys.argv[1:]
        parsed_config = conf_utils.parse_space_separated_args(space_args)
        # Convert to Hydra compatible format
        hydra_args = []
        for key, value in parsed_config.items():
            if value == "":
                hydra_args.append(f"{key}='{value}'")
            else:
                hydra_args.append(f"{key}={value}")

        # Replace sys.argv with Hydra-compatible format
        sys.argv = [sys.argv[0]] + hydra_args

    # 使用Hydra运行训练函数
    train()
