import os

import dataclasses
from datetime import datetime
from pprint import pformat
from torch.utils.data import IterableDataset
from omegaconf import OmegaConf

import deepspeed

# `import src.data` before other `src.` to avoid `common_io` import error
from ..data import collator, vocab_builder, tokenizer, read_dataset, dataset_iterable
from ..models import (
    convert_to_legacy_config,
    GraphGPTTaskModel,
    GraphGPTDenoisingRegressionDoubleHeadsModel,
)
from ..utils import (
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
from ..utils.log_eval_dump_utils import ft_evaluate as evaluate
from ..conf import (
    OptimizingStats,
    TrainingStats,
    LoaderStats,
    base_configs,
)
from .mode import TrainingMode

ModelEmaV3 = patch_utils.ModelEmaV3


class FinetuneMode(TrainingMode):
    """Strategy for supervised fine-tuning: epoch-level evaluation, separate
    train/valid/test datasets, FTSamplerConfig, layer freezing, eval_only
    and infer_only modes."""

    def __init__(self):
        # Mode-specific state (populated during setup)
        self._train_cfg = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.raw_dataset = None
        self.ft_sampler = None
        self.steps_per_epoch = None
        self.scheduler_conf = None
        self.collator_fn = None
        self.train_loader_for_eval = None
        self.valid_loader = None
        self.test_loader = None
        # Reference to train dataset for dict_bounds check in _create_model
        self._train_dataset_for_bounds = None

    @property
    def dict_models(self):
        return {
            "graphgpt": GraphGPTTaskModel,
            "graphgpt-denoise": GraphGPTDenoisingRegressionDoubleHeadsModel,
        }

    @property
    def skip_keys(self):
        return False

    def allow_resume(self):
        return not self._train_cfg.ft_eval.eval_only

    def allow_save_config(self):
        return not self._train_cfg.ft_eval.eval_only

    # ------------------------------------------------------------------ #
    #  update_config
    # ------------------------------------------------------------------ #

    def update_config(self, pipeline):
        cfg = pipeline.cfg
        cfg = base_configs.update_cfg_with_saved_cfg_yaml(cfg)
        pipeline.cfg = cfg
        base_configs.update_odps_cfg_for_ft_infer(cfg)
        base_configs.update_finetune_cfg(cfg)

        # Re-extract since cfg may have been replaced
        pipeline.token_cfg = cfg.tokenization
        pipeline.model_cfg = cfg.model
        pipeline.train_cfg = cfg.training
        pipeline.data_cfg = pipeline.token_cfg.data
        pipeline.sched_cfg = pipeline.train_cfg.schedule
        pipeline.optim_cfg = pipeline.train_cfg.optimizer

        train_cfg = pipeline.train_cfg
        data_cfg = pipeline.data_cfg
        self._train_cfg = train_cfg

        train_cfg.pretrain_mode = False
        data_cfg.return_valid_test = True
        data_cfg.odps.mode = "all"

        if pipeline.model_cfg.model_type == "graphgpt-denoise":
            train_cfg.finetune.task_ratio = 0.5

    # ------------------------------------------------------------------ #
    #  prepare_data
    # ------------------------------------------------------------------ #

    def prepare_data(self, pipeline):
        cfg = pipeline.cfg
        token_cfg = pipeline.token_cfg
        model_cfg = pipeline.model_cfg
        train_cfg = pipeline.train_cfg
        data_cfg = pipeline.data_cfg

        # 1.1 build tokenizer config
        tokenizer_config = conf_utils.convert_to_legacy_tokenization_config(cfg)
        if token_cfg.semantics.node.embed is None:
            tokenizer_config["semantics"]["node"].pop("embed", None)
            tokenizer_config["semantics"]["node"].pop("embed_dim", None)
        if token_cfg.semantics.edge.embed is None:
            tokenizer_config["semantics"]["edge"].pop("embed", None)
            tokenizer_config["semantics"]["edge"].pop("embed_dim", None)
        from pprint import pprint
        pprint(tokenizer_config)

        # 1.2 get graph datasets (train, valid, test, raw)
        self.train_dataset, self.valid_dataset, self.test_dataset, self.raw_dataset = (
            read_dataset(
                name=data_cfg.dataset,
                data_cfg=data_cfg,
                train_cfg=train_cfg,
                with_prob=False,
                true_valid=train_cfg.ft_eval.true_valid,
            )
        )
        self._train_dataset_for_bounds = self.train_dataset

        # inspect data points
        for dataset in [self.train_dataset, self.valid_dataset, self.test_dataset]:
            if isinstance(dataset, IterableDataset):
                print(next(iter(dataset)))
            else:
                idx = dataset.sampler[0]
                print(dataset[idx])

        # 1.3 build vocab
        vocab_builder.build_vocab(
            self.raw_dataset, tokenizer_config, cfg.training.distributed.rank
        )

        # 1.4 init tokenizer
        tokenizer_cls = getattr(tokenizer, tokenizer_config["tokenizer_class"])
        gtokenizer = tokenizer_cls(
            tokenizer_config,
            stack_method=model_cfg.graph_input.stack_method,
            loss_type=model_cfg.ft_head.loss_type,
            num_labels=model_cfg.ft_head.num_labels,
        )
        inspect_tokenization_results(self.train_dataset, gtokenizer)

        # 1.5 get train/valid/test sampler
        self.ft_sampler = loader_utils.FTSamplerConfig(
            train=loader_utils.SamplerConfig(ds=self.train_dataset),
            valid=loader_utils.SamplerConfig(ds=self.valid_dataset),
            test=loader_utils.SamplerConfig(ds=self.test_dataset),
        )
        self.steps_per_epoch = loader_utils.set_train_valid_test_sampler(
            self.ft_sampler, train_cfg
        )
        print(f"steps_per_epoch: {self.steps_per_epoch}")
        self.ft_sampler.enlarge_valid_test_samples(train_cfg.ft_eval.eval_only, 1)

        # update schedule
        samples_per_gpu = (
            (len(self.ft_sampler.train.sampler) if self.ft_sampler.train.sampler else self.ft_sampler.train.cnt)
            // pipeline.world_size
        )
        base_configs.update_ft_num_steps(train_cfg, samples_per_gpu)

        # 2.1 set model config
        pipeline.model_cfg = model_cfg = modules_utils.set_ft_model_config(
            cfg, gtokenizer
        )
        pipeline.config = convert_to_legacy_config(model_cfg)
        print(f"\nFinal model config for supervised task:\n{pformat(pipeline.config)}\n")

        # Store on pipeline
        pipeline.gtokenizer = gtokenizer
        pipeline.tokenizer_cls = tokenizer_cls
        pipeline.tokenizer_config = tokenizer_config

    # ------------------------------------------------------------------ #
    #  post_model_setup
    # ------------------------------------------------------------------ #

    def post_model_setup(self, pipeline):
        model = pipeline.model
        train_cfg = pipeline.train_cfg

        if train_cfg.finetune.freeze > -1:
            modules_utils.freeze_llama_layers(model, train_cfg.finetune.freeze)
        model.config.num_params = print_trainable_parameters(model)

        return False  # No early exit in FT mode

    # ------------------------------------------------------------------ #
    #  setup_optimizer
    # ------------------------------------------------------------------ #

    def setup_optimizer(self, pipeline):
        model = pipeline.model
        train_cfg = pipeline.train_cfg
        optim_cfg = pipeline.optim_cfg

        # 3.1 set main task, aux task ratio
        base_configs.set_finetune_cfg(train_cfg.finetune)

        # 3.2 create optimizer
        model_parameters = model.parameters()
        self.scheduler_conf = None
        if pipeline.use_deepspeed:
            (
                ds_config,
                non_ds_scheduler,
                self.scheduler_conf,
            ) = conf_utils.parse_deepspeed_config_for_ft(train_cfg, loss_utils)
            model, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model,
                model_parameters=model_parameters,
                lr_scheduler=non_ds_scheduler,
                config=ds_config,
                mpu=None,
                dist_init_required=False,
            )
            pipeline.opt_stats = OptimizingStats(optimizer, lr_scheduler)
        else:
            model, pipeline.opt_stats = opt_utils.initialize_optimizer(
                model=model,
                model_parameters=model_parameters,
                training=train_cfg,
                loss_utils=loss_utils,
            )
        pipeline.model = model
        pipeline.device = model.device
        print(f"optimizer: {pipeline.opt_stats.optimizer}")
        print(f"[{datetime.now()}] Finish -> 3. set optimizer")

        pipeline.ema_stats.init_ema(model, ModelEmaV3, optim_cfg.ema_decay)
        pipeline.ema_stats.ema2device(pipeline.device, pipeline.ema_cfg.use_ema)

    # ------------------------------------------------------------------ #
    #  setup_training
    # ------------------------------------------------------------------ #

    def setup_training(self, pipeline):
        cfg = pipeline.cfg
        model = pipeline.model
        train_cfg = pipeline.train_cfg
        sched_cfg = pipeline.sched_cfg
        output_dir = pipeline.output_dir
        rank = pipeline.rank

        use_tb_writer = False

        # 4.2 init log config
        (
            _,
            ep_init,
            j_init,
            ls_log,
            ls_result,
            ls_loss,
        ) = conf_utils.init_log_conf_for_ft(
            misc_utils=misc_utils,
            pretrain_cpt=pipeline.pretrain_cpt,
            output_dir=output_dir,
            steps_per_epoch=self.steps_per_epoch,
            eval_only=train_cfg.ft_eval.eval_only,
        )
        print(f"[{datetime.now()}] Training start with j_init {j_init} and ep_init {ep_init} ...")

        # 4.3 init collator
        self.collator_fn = collator.DataCollatorForGST(
            tokenizer=pipeline.gtokenizer,
            max_length=train_cfg.max_length,
            pad_to_multiple_of=train_cfg.pad_to_multiple_of,
            return_tensors="pt",
            is_training=False,
        )
        print(f"[{datetime.now()}] Finish -> 4.3 init collator")

        # 4.41 set-up eval loaders
        train_cfg.num_workers_eval = min(train_cfg.num_workers, 16)
        (
            self.train_loader_for_eval,
            self.valid_loader,
            self.test_loader,
        ) = loader_utils.get_eval_loader(self.ft_sampler, train_cfg, self.collator_fn)

        # 4.42 TB writer
        pipeline.tb_writer = log_eval_dump_utils.ft_dump_cfg_and_init_tb(
            model,
            pipeline.use_deepspeed,
            use_tb_writer,
            output_dir,
            train_cfg.ft_eval.eval_only,
            self.scheduler_conf,
        )

        # TrainingStats
        pipeline.train_stats = TrainingStats(
            device=pipeline.device,
            has_embeds_input=pipeline.model_cfg.graph_input.embed_dim > 0,
            use_deepspeed=pipeline.use_deepspeed,
            epoch_start=ep_init,
            j=j_init,
            ls_log=ls_log,
            ls_loss=ls_loss,
            ls_result=ls_result,
        )

        # Pre-training evaluation
        if not train_cfg.ft_eval.eval_only:
            print(f"[{datetime.now()}] Eval before training starts ...")
            val_loss, val_cls_metrics, val_ogb_eval_res, val_triplet = evaluate(
                model, self.valid_loader, cfg, "valid"
            )
            print(
                f"[{datetime.now()}] tr_loss: {val_loss}\n"
                f"tr_cls_metrics: {val_cls_metrics.results_in_details()}\n"
                f"tr_ogb_eval_res: {val_ogb_eval_res}, tr_triplet: {val_triplet}"
            )
            if rank == 0:
                misc_utils.save_all(
                    output_dir,
                    model,
                    epoch=-1,
                    save_model=False,
                    val_dict=val_triplet if train_cfg.ft_eval.save_pred else None,
                )
            pipeline.ema_stats.ema_best_res = val_ogb_eval_res

        if train_cfg.ft_eval.eval_only:
            ep_init = max(ep_init - 1, train_cfg.schedule.epochs - 1)
            pipeline.train_stats.epoch_start = ep_init
            print(
                f"[{datetime.now()}] EVAL only mode, ep_init: {ep_init}, "
                f"epochs: {sched_cfg.epochs}!"
            )

    # ------------------------------------------------------------------ #
    #  run_training
    # ------------------------------------------------------------------ #

    def run_training(self, pipeline):
        cfg = pipeline.cfg
        model = pipeline.model
        train_cfg = pipeline.train_cfg
        sched_cfg = pipeline.sched_cfg
        train_stats = pipeline.train_stats
        opt_stats = pipeline.opt_stats
        ema_stats = pipeline.ema_stats
        tb_writer = pipeline.tb_writer
        output_dir = pipeline.output_dir
        rank = pipeline.rank
        data_cfg = pipeline.data_cfg

        model.train()
        if not train_cfg.ft_eval.eval_only:
            OmegaConf.save(config=cfg, f=os.path.join(output_dir, "config.yaml"))

        with pipeline.tmp_env:
            for epoch in range(train_stats.epoch_start, sched_cfg.epochs):
                train_stats.epoch = epoch
                loader_stats = LoaderStats(
                    train_loader_for_eval=self.train_loader_for_eval,
                    valid_loader=self.valid_loader,
                    test_loader=self.test_loader,
                )
                if not train_cfg.ft_eval.eval_only:
                    train_loader = (
                        loader_utils.initialize_ft_train_loader_at_epoch_start(
                            self.train_dataset,
                            train_cfg,
                            train_stats,
                            self.ft_sampler,
                            self.collator_fn,
                        )
                    )
                    loader_stats = dataclasses.replace(
                        loader_stats, train_loader=train_loader
                    )
                    train_stats.t_start = datetime.now()

                    for i, data in enumerate(loader_stats.train_loader):
                        train_stats.i = i
                        training_utils.ft_batch_training(
                            data,
                            model,
                            pipeline.model_cfg.ft_head,
                            train_cfg,
                            train_stats,
                            opt_stats,
                        )
                        ema_stats.update_ema(model, step=train_stats.j, ft=True)
                        if train_stats.j % sched_cfg.logging_steps == 0:
                            log_eval_dump_utils.log_ft_training_stats(
                                train_cfg, train_stats, tb_writer
                            )
                        train_stats.j += 1
                else:
                    # eval_only mode: load from ckp and then eval
                    ckp = os.path.join(pipeline.pretrain_cpt, f"epoch_{epoch}")
                    if os.path.exists(ckp):
                        loader_utils.load_from_ckp_with_try(
                            model.module,
                            ckp,
                            skip_keys=False,
                            use_ema=ema_stats.ema_cfg.use_ema,
                        )
                    else:
                        print(f"ckp {ckp} doesn't exists, skip it!")
                    ema_stats.model_ema = None

                if train_cfg.ft_eval.infer_only:
                    writer = dataset_iterable.get_odps_writer(
                        table_name=data_cfg.odps.outputs, slice_id=rank
                    )
                    misc_utils.dump_results(
                        model=model,
                        loader=self.test_loader,
                        device=pipeline.device,
                        writer=writer,
                        slice_id=rank,
                    )
                    writer.close()

                if (epoch + 1) % train_cfg.ft_eval.epoch_per_eval == 0 and (
                    not train_cfg.ft_eval.infer_only
                ):
                    log_eval_dump_utils.log_dump_ft_training_stats(
                        model,
                        cfg,
                        self.ft_sampler,
                        train_stats,
                        opt_stats,
                        loader_stats,
                        ema_stats,
                        tb_writer,
                    )
