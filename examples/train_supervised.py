import os

import dataclasses
import multiprocessing as mp
import deepspeed
from datetime import datetime
from pprint import pprint, pformat
from torch.utils.data import IterableDataset

import sys
sys.path.insert(0, ".")
# put below `src.data` import above other `src.` to avoid `common_io` import error
from src.data import collator, vocab_builder, tokenizer, read_dataset, dataset_iterable
from src.models import (
    convert_to_legacy_config,
    GraphGPTTaskModel,
    GraphGPTDenoisingRegressionDoubleHeadsModel,
)
from src.utils import (
    patch_utils,
    conf_utils,
    loader_utils,
    log_eval_dump_utils,
    modules_utils,
    misc_utils,
    loss_utils,
    print_trainable_parameters,
    inspect_tokenization_results,
    opt_utils,
    training_utils,
)
from src.utils.log_eval_dump_utils import ft_evaluate as evaluate
from src.conf import (
    Config,
    OptimizingStats,
    TrainingStats,
    EMAConfig,
    EMAStats,
    LoaderStats,
    base_configs,
)

# Hydra imports
import hydra
from omegaconf import OmegaConf

ModelEmaV3 = patch_utils.ModelEmaV3

dict_models = {
    "graphgpt": GraphGPTTaskModel,
    "graphgpt-denoise": GraphGPTDenoisingRegressionDoubleHeadsModel,
}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: Config):
    cfg = base_configs.update_cfg_with_saved_cfg_yaml(cfg)
    base_configs.update_odps_cfg_for_ft_infer(cfg)
    base_configs.update_finetune_cfg(cfg)
    # extract main config component
    token_cfg = cfg.tokenization
    model_cfg = cfg.model
    train_cfg = cfg.training
    # extract sub-configs
    data_cfg = token_cfg.data
    sched_cfg = train_cfg.schedule
    optim_cfg = train_cfg.optimizer
    train_cfg.pretrain_mode = False
    data_cfg.return_valid_test = True
    data_cfg.odps.mode = "all"

    # extrac params from configs
    pretrain_cpt = train_cfg.pretrain_cpt
    output_dir = train_cfg.output_dir

    # set some configs
    if model_cfg.model_type == "graphgpt-denoise":
        train_cfg.finetune.task_ratio = 0.5

    use_tb_writer = False
    ema_cfg = EMAConfig(
        use_ema=optim_cfg.use_ema,
        ema_file="model_ema.pt",
        ema_file_best="model_ema_best.pt",
    )
    ema_stats = EMAStats(ema_cfg=ema_cfg)

    train_cfg.use_deepspeed = use_deepspeed = len(train_cfg.deepspeed_conf_file) > 0
    if os.path.exists(os.path.join(output_dir, "log.csv")):
        print(
            f"log file\n{os.path.join(output_dir, 'log.csv')}\nexists, resume training from\n{output_dir}\ninstead of initializing from pre-train ckp\n{pretrain_cpt}"
        )
        pretrain_cpt = output_dir
    # 0. set-up distributed training
    tmp_env = misc_utils.set_dist_env(train_cfg)
    world_size = train_cfg.distributed.world_size
    rank = train_cfg.distributed.rank

    # 1. prepare data & tokenizer
    # 1.1 read configuration
    base_configs.init_stacked_feat(cfg)
    base_configs.init_embed_dim(cfg)
    base_configs.sync_config(cfg)

    tokenizer_config = conf_utils.convert_to_legacy_tokenization_config(cfg)
    if token_cfg.semantics.node.embed is None:
        tokenizer_config["semantics"]["node"].pop("embed", None)
        tokenizer_config["semantics"]["node"].pop("embed_dim", None)
    if token_cfg.semantics.edge.embed is None:
        tokenizer_config["semantics"]["edge"].pop("embed", None)
        tokenizer_config["semantics"]["edge"].pop("embed_dim", None)
    pprint(tokenizer_config)

    # 1.2 get graph dataset
    train_dataset, valid_dataset, test_dataset, raw_dataset = read_dataset(
        name=data_cfg.dataset,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        with_prob=False,
        true_valid=train_cfg.ft_eval.true_valid,
    )
    # inspect data points
    for dataset in [train_dataset, valid_dataset, test_dataset]:
        if isinstance(dataset, IterableDataset):
            print(next(iter(dataset)))
        else:
            idx = dataset.sampler[0]
            print(dataset[idx])

    # 1.3 build vocab if needed
    vocab_builder.build_vocab(raw_dataset, tokenizer_config, cfg.training.distributed.rank)

    # 1.4 init tokenizer from the tokenization config
    tokenizer_cls = getattr(tokenizer, tokenizer_config["tokenizer_class"])
    gtokenizer = tokenizer_cls(
        tokenizer_config,
        stack_method=model_cfg.graph_input.stack_method,
        loss_type=model_cfg.ft_head.loss_type,
        num_labels=model_cfg.ft_head.num_labels,
    )  # loss_type & num_labels -> kwargs

    inspect_tokenization_results(train_dataset, gtokenizer)

    # 1.5 get train/valid/test sampler
    ft_sampler = loader_utils.FTSamplerConfig(
        train=loader_utils.SamplerConfig(ds=train_dataset),
        valid=loader_utils.SamplerConfig(ds=valid_dataset),
        test=loader_utils.SamplerConfig(ds=test_dataset),
    )
    steps_per_epoch = loader_utils.set_train_valid_test_sampler(ft_sampler, train_cfg)
    print(f"steps_per_epoch: {steps_per_epoch}")

    ft_sampler.enlarge_valid_test_samples(train_cfg.ft_eval.eval_only, 1)
    # due to `drop_last` in train_loader, use //
    samples_per_gpu = (len(ft_sampler.train.sampler) if ft_sampler.train.sampler else ft_sampler.train.cnt) // world_size
    base_configs.update_ft_num_steps(train_cfg, samples_per_gpu)

    # 2. set model
    # 2.1 init model config
    model_cfg = modules_utils.set_ft_model_config(cfg, gtokenizer)
    config = convert_to_legacy_config(model_cfg)
    print(f"\nFinal model config for supervised task:\n{pformat(config)}\n")
    # 2.2 create model
    if use_deepspeed:
        deepspeed.init_distributed(
            dist_backend="nccl", rank=rank, world_size=world_size
        )
    model = dict_models[model_cfg.model_type](config)
    if hasattr(ft_sampler.train.ds, "dict_bounds"):  # for PCQM4M-v2 dataset ONLY
        model.dict_bounds = ft_sampler.train.ds.dict_bounds
    model.gradient_checkpointing_enable()
    # silence the warnings. Please re-enable for inference!
    model.config.use_cache = False
    if train_cfg.finetune.freeze > -1:  # 0->freeze embedding; 1->embed+1st layer
        modules_utils.freeze_llama_layers(model, train_cfg.finetune.freeze)
    model.config.num_params = print_trainable_parameters(model)
    # 2.3 load from ckp IF provided existing ckp and NOT resume from the ckp
    model = loader_utils.load_from_ckp(
        misc_utils=misc_utils,
        pretrain_cpt=pretrain_cpt,
        output_dir=output_dir,
        model=model,
        config=config,
        skip_keys=False,
    )
    print(model)

    # 3. Setup training::optimizer (load optimization config if given)
    # 3.1 set main task, aux task ratio
    base_configs.set_finetune_cfg(train_cfg.finetune)
    # 3.2 Create optimizer (load optimization config if given)
    # obtain layerwise lr
    model_parameters = model.parameters()
    # model_parameters = loss_utils.get_layerwise_param_groups(model, lr, 0.95)
    if use_deepspeed:
        (
            ds_config,
            non_ds_scheduler,
            scheduler_conf,
        ) = conf_utils.parse_deepspeed_config_for_ft(train_cfg, loss_utils)
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            lr_scheduler=non_ds_scheduler,
            config=ds_config,
            mpu=None,
            dist_init_required=False,
        )
        opt_stats = OptimizingStats(optimizer, lr_scheduler)
    else:
        model, opt_stats = opt_utils.initialize_optimizer(
            model=model,
            model_parameters=model_parameters,
            training=train_cfg,
            loss_utils=loss_utils,
        )
    device = model.device
    print(f"optimizer: {opt_stats.optimizer}")
    print(f"[{datetime.now()}] Finish -> 3. set optimizer")
    ema_stats.init_ema(model, ModelEmaV3, optim_cfg.ema_decay)
    ema_stats.ema2device(device, ema_cfg.use_ema)

    # 4. SET initial status
    # 4.1 Resuming: load model parameters and optimizer stats from ckp from current ckp
    if (len(pretrain_cpt) > 0) and (pretrain_cpt == output_dir) and (not train_cfg.ft_eval.eval_only):
        ckp, _ = misc_utils.get_latest_ckp(pretrain_cpt)
        if use_deepspeed:
            print(f"Resuming: Load weights from {ckp} with deepspeed API.")
            model.load_checkpoint(ckp)
        else:
            misc_utils.load_ddp_ckp(
                ckp,
                model=model,
                optimizer=opt_stats.optimizer,
                lr_scheduler=opt_stats.lr_scheduler,
            )
        print(
            f"After loading weights from ckp:\n{model.module.config}\n"
            f"num_labels: {model.module.num_labels}\nmodel-type: {model.module.dtype}\n\n{model.module}"
        )
        ema_stats.load_ema_ckp(output_dir)

    if (train_cfg.distributed.rank == 0) and (not train_cfg.ft_eval.eval_only):
        try:
            model.module.config.save_pretrained(output_dir)
        except AttributeError:
            print("In local test setting!!!\n" * 5)
            model.config.save_pretrained(output_dir)
        print(
            f"[{datetime.now()}] Finish -> Dump model config to `{output_dir}/config.json`"
        )

    # 4.2 set initial condition of training, either resuming from ckp or starting from scratch
    (
        _,
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
        eval_only=train_cfg.ft_eval.eval_only,
    )

    # 4.3 init collator
    collator_fn = collator.DataCollatorForGST(
        tokenizer=gtokenizer,
        max_length=train_cfg.max_length,
        pad_to_multiple_of=train_cfg.pad_to_multiple_of,
        return_tensors="pt",
        is_training=False,
    )
    print(f"[{datetime.now()}] Finish -> 4.3 init collator")
    # 4.4 set-up loader
    # 4.41 set-up valid/test loader and evaluate before training
    train_cfg.num_workers_eval = min(train_cfg.num_workers, 16)
    train_loader_for_eval, valid_loader, test_loader = loader_utils.get_eval_loader(ft_sampler, train_cfg, collator_fn)

    # 4.42 dump ds config and initialize tensorboard in `worker 0`
    tb_writer = log_eval_dump_utils.ft_dump_cfg_and_init_tb(
        model, use_deepspeed, use_tb_writer, output_dir, train_cfg.ft_eval.eval_only, scheduler_conf
    )

    # 4. Training & Inferring ...
    print(f"[{datetime.now()}] Training start with j_init {j_init} and ep_init {ep_init} ...")
    j = j_init

    train_stats = TrainingStats(
        device=device,
        has_embeds_input=model_cfg.graph_input.embed_dim > 0,
        use_deepspeed=use_deepspeed,
        epoch_start=ep_init,
        j=j,
        ls_log=ls_log,
        ls_loss=ls_loss,
        ls_result=ls_result,
    )

    if not train_cfg.ft_eval.eval_only:
        print(f"[{datetime.now()}] Eval before training starts ...")
        val_loss, val_cls_metrics, val_ogb_eval_res, val_triplet = evaluate(
            model, valid_loader, cfg, "valid"
        )
        print(
            f"[{datetime.now()}] tr_loss: {val_loss}\ntr_cls_metrics: {val_cls_metrics.results_in_details()}\n"
            f"tr_ogb_eval_res: {val_ogb_eval_res}, tr_triplet: {val_triplet}"
        )
        if rank == 0:
            misc_utils.save_all(
                output_dir,
                model,
                epoch=-1,
                save_model=False,
                val_dict=val_triplet if train_cfg.ft_eval.save_pred else None
            )
        ema_stats.ema_best_res = val_ogb_eval_res
    if train_cfg.ft_eval.eval_only:
        ep_init = max(ep_init - 1, train_cfg.schedule.epochs - 1)
        train_stats.epoch_start = ep_init
        print(
            f"[{datetime.now()}] EVAL only mode, ep_init: {ep_init}, epochs: {sched_cfg.epochs}!"
        )
    model.train()
    if not train_cfg.ft_eval.eval_only:
        OmegaConf.save(config=cfg, f=os.path.join(output_dir, "config.yaml"))
    with tmp_env:  # must be in the outerest loop, otherwise error occurs
        for epoch in range(train_stats.epoch_start, sched_cfg.epochs):
            train_stats.epoch = epoch
            loader_stats = LoaderStats(
                train_loader_for_eval=train_loader_for_eval,
                valid_loader=valid_loader,
                test_loader=test_loader
            )
            if not train_cfg.ft_eval.eval_only:
                train_loader = loader_utils.initialize_ft_train_loader_at_epoch_start(
                    train_dataset, train_cfg, train_stats, ft_sampler, collator_fn
                )
                loader_stats = dataclasses.replace(loader_stats, train_loader=train_loader)
                # print(f"Top 10 samples' idx:\n{train_loader.sampler[:10]}")
                train_stats.t_start = datetime.now()

                for i, data in enumerate(loader_stats.train_loader):
                    train_stats.i = i
                    # Iterate in batches over the training dataset.
                    training_utils.ft_batch_training(
                        data, model, model_cfg.ft_head, train_cfg, train_stats, opt_stats
                    )
                    ema_stats.update_ema(model, step=train_stats.j, ft=True)
                    if train_stats.j % sched_cfg.logging_steps == 0:
                        log_eval_dump_utils.log_ft_training_stats(train_cfg, train_stats, tb_writer)
                    train_stats.j += 1
            else:  # `eval_only` mode, load from ckp and then eval
                ckp = os.path.join(pretrain_cpt, f"epoch_{epoch}")
                if os.path.exists(ckp):
                    loader_utils.load_from_ckp_with_try(
                        model.module, ckp, skip_keys=False, use_ema=ema_stats.ema_cfg.use_ema
                    )
                else:
                    print(f"ckp {ckp} doesn't exists, skip it!")
                ema_stats.model_ema = None
            if train_cfg.ft_eval.infer_only:  # INFER in ODPS
                writer = dataset_iterable.get_odps_writer(table_name=data_cfg.odps.outputs, slice_id=rank)
                misc_utils.dump_results(
                    model=model,
                    loader=test_loader,
                    device=device,
                    writer=writer,
                    slice_id=rank,
                )
                writer.close()
            if (epoch + 1) % train_cfg.ft_eval.epoch_per_eval == 0 and (not train_cfg.ft_eval.infer_only):
                log_eval_dump_utils.log_dump_ft_training_stats(
                    model, cfg, ft_sampler, train_stats, opt_stats, loader_stats, ema_stats, tb_writer
                )

    tb_writer.close() if tb_writer is not None else None
    if not train_cfg.ft_eval.eval_only:
        OmegaConf.save(config=cfg, f=os.path.join(output_dir, "config.yaml"))


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

    # Use Hydra to train
    train()
