import os

from datetime import datetime
from pprint import pprint, pformat
from torch.utils.data import DataLoader, IterableDataset
from deepspeed.profiling.flops_profiler import FlopsProfiler
from omegaconf import OmegaConf

import deepspeed

# `import src.data` before other `src.` to avoid `common_io` import error
from ..data import (
    collator,
    vocab_builder,
    tokenizer,
    read_dataset,
    OdpsTableIterableDataset,
)
from ..models import (
    convert_to_legacy_config,
    GraphGPTPretrainBase,
    GraphGPTPosPred,
)
from ..utils import (
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
from ..utils.log_eval_dump_utils import evaluate, evaluate_generation, eval_pt_gen_only
from ..conf import (
    OptimizingStats,
    TrainingStats,
    OdpsStats,
    LoaderStats,
    base_configs,
)
from .mode import TrainingMode


class PretrainMode(TrainingMode):
    """Strategy for pre-training: step-level checkpointing, single dataset,
    PTSamplerConfig, token packing, generation evaluation."""

    def __init__(self):
        # Mode-specific state (populated during setup)
        self.dataset = None
        self.raw_dataset = None
        self.pt_sampler = None
        self.tokens_per_sample = 0
        self.reset_samples_per_epoch = False
        self.task_type = None
        self.batch_size = None
        self.prof = None  # FlopsProfiler (DDP only)
        self.collator_fn = None
        self.train_loader = None
        self.valid_loader = None
        self.valgen_loader = None
        self.test_loader = None
        # Reference to train dataset for dict_bounds check in _create_model
        self._train_dataset_for_bounds = None

    @property
    def dict_models(self):
        return {
            "graphgpt": GraphGPTPretrainBase,
            "graphgpt-pos": GraphGPTPosPred,
        }

    # ------------------------------------------------------------------ #
    #  update_config
    # ------------------------------------------------------------------ #

    def update_config(self, pipeline):
        train_cfg = pipeline.train_cfg
        optim_cfg = pipeline.optim_cfg
        sched_cfg = pipeline.sched_cfg

        train_cfg.pretrain_mode = True
        train_cfg.do_valid = train_cfg.valid_percent > 0
        train_cfg.do_test = optim_cfg.use_ema
        base_configs.update_odps_cfg_from_token_cfg(pipeline.cfg, "train")

        # NOTE: min_lr is set later in prepare_data() after pipeline.use_deepspeed is determined

    # ------------------------------------------------------------------ #
    #  prepare_data
    # ------------------------------------------------------------------ #

    def prepare_data(self, pipeline):
        cfg = pipeline.cfg
        token_cfg = pipeline.token_cfg
        model_cfg = pipeline.model_cfg
        train_cfg = pipeline.train_cfg
        data_cfg = pipeline.data_cfg
        sched_cfg = pipeline.sched_cfg
        optim_cfg = pipeline.optim_cfg

        # min_lr depends on use_deepspeed which is set by _setup_deepspeed_flag()
        # before prepare_data() is called, so it's safe to compute here.
        optim_cfg.min_lr = optim_cfg.lr * 0.1 if pipeline.use_deepspeed else 0

        self.task_type = task_type = train_cfg.task_type
        self.batch_size = batch_size = train_cfg.batch_size

        # Steps per saving
        sched_cfg.steps_per_saving = sched_cfg.samples_per_saving // (
            pipeline.world_size * batch_size
        )

        # 1.1 build tokenizer config
        tokenizer_config = conf_utils.convert_to_legacy_tokenization_config(cfg)
        pprint(tokenizer_config)
        model_cfg.pt_head.smtp_inside = False
        tmp_method = tokenizer_config.get("pretrain_mlm", {}).get("method", None)
        if tmp_method == "inside_model":
            model_cfg.pt_head.smtp_inside = True
            assert tokenizer_config["task_type"] in {"pretrain-mlm", "pretrain-smtp"}
            tokenizer_config["task_type"] = task_type = "pretrain-smtp"
            self.task_type = task_type
            print(f"set task_type = {task_type} for {tmp_method}")

        # 1.2 read dataset
        dataset, raw_dataset = read_dataset(
            name=data_cfg.dataset, data_cfg=data_cfg, train_cfg=train_cfg
        )
        self.dataset = dataset
        self.raw_dataset = raw_dataset
        self._train_dataset_for_bounds = dataset
        self.reset_samples_per_epoch = (
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
        self.pt_sampler = loader_utils.get_pt_train_valid_test_sampler(
            train_dataset, train_cfg, task_type, data_cfg, read_dataset,
        )

        # 1.4 build vocab
        vocab_builder.build_vocab(
            raw_dataset, tokenizer_config, cfg.training.distributed.rank
        )

        # 1.5 init tokenizer
        tokenizer_cls = getattr(tokenizer, tokenizer_config["tokenizer_class"])
        gtokenizer = tokenizer_cls(
            tokenizer_config,
            add_eos=token_cfg.add_eos,
            stack_method=model_cfg.graph_input.stack_method,
            train_cfg=train_cfg,
        )
        base_configs.update_generation_cfg(cfg, gtokenizer)

        # 1.51 token packing
        if train_cfg.pack_tokens > 0:
            gtokenizer.mpe = model_cfg.max_position_embeddings
            gtokenizer.dataset = train_dataset
            gtokenizer.sampler = (
                tuple(self.pt_sampler.train_sampler)
                if self.pt_sampler.train_sampler is not None
                else None
            )
            gtokenizer.random_ratio = train_cfg.pack_tokens
            tokens_per_sample = model_cfg.max_position_embeddings
        else:
            tokens_per_sample = misc_utils.estimate_tokens_per_sample(
                gtokenizer,
                train_dataset,
                self.pt_sampler.train_sampler,
                model_cfg.max_position_embeddings,
                train_cfg.distributed.world_size,
                train_cfg.tot_samples
                if not (train_cfg.pt_eval_only or train_cfg.do_infer)
                else 100,
            )
        tokens_per_sample = (
            tokens_per_sample // 2
            if task_type == "pretrain-euler"
            else tokens_per_sample
        )
        self.tokens_per_sample = tokens_per_sample
        print(f"\n[{datetime.now()}] tokens_per_sample: {tokens_per_sample}")

        # 1.52 inspect tokenization results
        inspect_tokenization_results(dataset, gtokenizer)
        # 1.53 re-init to avoid pickle error
        gtokenizer.dataset = train_dataset if train_cfg.pack_tokens > 0 else None

        # 1.54 update training schedule
        base_configs.update_num_steps(
            sched_cfg, tokens_per_sample, batch_size, pipeline.world_size
        )
        samples_per_gpu = (
            len(self.pt_sampler.train_sampler)
            if self.pt_sampler.train_sampler
            else self.pt_sampler.train_cnt / pipeline.world_size
        )
        base_configs.update_epochs(
            sched_cfg, tokens_per_sample, samples_per_gpu, pipeline.world_size
        )
        base_configs.print_stats(sched_cfg)

        # 2.1 set model config
        pipeline.model_cfg = model_cfg = modules_utils.set_model_config(cfg, gtokenizer)
        pipeline.config = self.config = convert_to_legacy_config(model_cfg)
        print(pipeline.config)

        # Store on pipeline for downstream phases
        pipeline.gtokenizer = gtokenizer
        pipeline.tokenizer_cls = tokenizer_cls
        pipeline.tokenizer_config = tokenizer_config

    # ------------------------------------------------------------------ #
    #  post_model_setup
    # ------------------------------------------------------------------ #

    def post_model_setup(self, pipeline):
        model = pipeline.model
        cfg = pipeline.cfg
        train_cfg = pipeline.train_cfg

        print_trainable_parameters(model)

        # EVAL ONLY MODE
        if train_cfg.pt_eval_only:
            eval_pt_gen_only(
                model,
                cfg,
                collator.DataCollatorForGST,
                pipeline.tokenizer_cls,
                pipeline.tokenizer_config,
                self.pt_sampler,
                self.dataset,
            )
            return True

        # INFER ONLY MODE
        if train_cfg.do_infer:
            train_cfg.pretrain_mlm.params.umr_clip = [1, 1]
            log_eval_dump_utils.pt_infer_only(
                model,
                cfg,
                collator.DataCollatorForGST,
                pipeline.tokenizer_cls,
                pipeline.tokenizer_config,
                self.pt_sampler,
            )
            return True

        return False

    # ------------------------------------------------------------------ #
    #  setup_optimizer
    # ------------------------------------------------------------------ #

    def setup_optimizer(self, pipeline):
        model = pipeline.model
        train_cfg = pipeline.train_cfg

        model_parameters = model.parameters()
        if pipeline.use_deepspeed:
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
            pipeline.opt_stats = OptimizingStats(optimizer, lr_scheduler)
            self.prof = None
        else:
            self.prof = FlopsProfiler(model)
            model, pipeline.opt_stats = opt_utils.initialize_optimizer(
                model=model,
                model_parameters=model_parameters,
                training=train_cfg,
                loss_utils=loss_utils,
            )
        pipeline.model = model
        pipeline.device = model.device
        print(f"[{datetime.now()}] Finish -> 3. set optimizer")

        pipeline.ema_stats.init_ema(model)

    # ------------------------------------------------------------------ #
    #  setup_training
    # ------------------------------------------------------------------ #

    def setup_training(self, pipeline):
        cfg = pipeline.cfg
        model = pipeline.model
        model_cfg = pipeline.model_cfg
        train_cfg = pipeline.train_cfg
        sched_cfg = pipeline.sched_cfg
        output_dir = pipeline.output_dir

        use_tb_writer = False

        # 4.2 init log config
        _, ckp_init, j_init, ls_log, ls_result = conf_utils.init_log_conf(
            misc_utils=misc_utils,
            pretrain_cpt=pipeline.pretrain_cpt,
            output_dir=output_dir,
            steps_per_saving=sched_cfg.steps_per_saving,
        )

        # 4.3 init collator
        self.collator_fn = collator.DataCollatorForGST(
            tokenizer=pipeline.gtokenizer,
            max_length=model_cfg.max_position_embeddings,
            pad_to_multiple_of=train_cfg.pad_to_multiple_of,
            return_tensors="pt",
        )
        print(f"[{datetime.now()}] Finish -> 4.3 init collator")

        # 4.41 set-up valid/test loader and evaluate before training
        self.valid_loader, self.valgen_loader = loader_utils.initialize_pt_valid_loader(
            self.dataset,
            cfg,
            self.pt_sampler,
            pipeline.tokenizer_config,
            pipeline.tokenizer_cls,
            collator.DataCollatorForGST,
        )
        evaluate(model, self.valid_loader, "valid", train_cfg.do_valid)
        evaluate_generation(
            model,
            self.valgen_loader,
            "valid",
            train_cfg.do_valid and train_cfg.do_generation,
            cfg,
        )
        self.test_loader = loader_utils.initialize_pt_test_loader(
            cfg,
            self.pt_sampler,
            pipeline.tokenizer_cls,
            pipeline.tokenizer_config,
            collator.DataCollatorForGST,
        )
        acc, triplet = evaluate(model, self.test_loader, "test", train_cfg.do_test)
        pipeline.ema_stats.ema_best_res = {"loss": acc} if acc else None

        # 4.42 dump ds config and init TB
        pipeline.tb_writer = log_eval_dump_utils.pt_dump_cfg_and_init_tb(
            model, pipeline.use_deepspeed, use_tb_writer, output_dir
        )

        # 4.43 reset train sampler
        self.pt_sampler = loader_utils.reset_pt_train_sampler(
            self.reset_samples_per_epoch,
            self.task_type,
            self.dataset,
            train_cfg,
            self.pt_sampler,
        )

        # 4.44 set-up train loader
        self.train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.pt_sampler.train_shuffle,
            sampler=self.pt_sampler.train_sampler,
            num_workers=train_cfg.num_workers,
            collate_fn=self.collator_fn,
            worker_init_fn=worker_init_fn_seed,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4,
        )

        # TrainingStats
        pipeline.train_stats = TrainingStats(
            device=pipeline.device,
            has_embeds_input=model_cfg.graph_input.embed_dim > 0,
            use_deepspeed=pipeline.use_deepspeed,
            ls_log=ls_log,
            ls_result=ls_result,
            reset_samples_per_epoch=self.reset_samples_per_epoch,
            tokens_per_sample=self.tokens_per_sample,
            ckp=ckp_init,
            j=j_init,
            odps_stats=OdpsStats(
                is_odps_table_ds=isinstance(self.dataset, OdpsTableIterableDataset),
                tables=pipeline.data_cfg.odps.tables,
            ),
        )
        loader_utils.init_odps_ds_stats(train_cfg, pipeline.train_stats, self.pt_sampler)

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

        j_init = train_stats.j

        print(f"[{datetime.now()}] 5. Training start ...")
        model.train()
        if not pipeline.use_deepspeed:
            self.prof.start_profile()
        OmegaConf.save(config=cfg, f=os.path.join(output_dir, "config.yaml"))

        for epoch in range(train_stats.epoch_start, sched_cfg.epochs):
            train_stats.epoch = epoch
            train_loader = (
                loader_utils.initialize_train_loader_at_epoch_start(
                    self.dataset,
                    train_cfg,
                    train_stats,
                    self.pt_sampler,
                    self.collator_fn,
                    OdpsTableIterableDataset,
                )
                or self.train_loader
            )
            loader_stats = LoaderStats(
                train_loader=train_loader,
                valid_loader=self.valid_loader,
                valgen_loader=self.valgen_loader,
                test_loader=self.test_loader,
            )
            train_stats.t_start = datetime.now()

            with pipeline.tmp_env:
                for i, data in enumerate(
                    loader_stats.train_loader, train_stats.i_local
                ):
                    train_stats.i = i
                    training_utils.batch_training(
                        data, model, train_cfg, train_stats, opt_stats
                    )
                    ema_stats.update_ema(model, step=train_stats.j)
                    if train_stats.j % sched_cfg.logging_steps == 0:
                        log_eval_dump_utils.log_pt_training_stats(
                            train_cfg,
                            train_stats,
                            opt_stats,
                            prof=self.prof,
                            tb_writer=tb_writer,
                        )

                    if (
                        (train_stats.j % sched_cfg.steps_per_saving == 0)
                        and (train_stats.j > j_init)
                    ) or (train_stats.j == sched_cfg.total_num_steps):
                        log_eval_dump_utils.log_dump_pt_training_stats(
                            model,
                            cfg,
                            train_stats,
                            opt_stats,
                            loader_stats,
                            ema_stats,
                            tb_writer,
                        )

                    if train_stats.j == sched_cfg.total_num_steps:
                        print(
                            f"Final step reached: {sched_cfg.total_num_steps}, "
                            f"break inner loop!"
                        )
                        break
                    train_stats.j += 1

            if train_stats.j >= sched_cfg.total_num_steps:
                print(
                    f"Final step reached: {sched_cfg.total_num_steps}, "
                    f"break outer loop!"
                )
                break

        if not pipeline.use_deepspeed:
            self.prof.end_profile()
