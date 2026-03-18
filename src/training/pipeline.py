import os
import sys
import multiprocessing as mp

import deepspeed
from datetime import datetime
from omegaconf import OmegaConf

from ..conf import Config, EMAConfig, EMAStats
from ..utils import misc_utils, loader_utils
from ..conf import base_configs
from .mode import TrainingMode


class TrainingPipeline:
    """Unified training pipeline that orchestrates shared setup phases
    and delegates mode-specific behavior to a TrainingMode strategy."""

    def __init__(self, cfg: Config, mode: TrainingMode):
        self.cfg = cfg
        self.mode = mode

        # Config components (set by _extract_config)
        self.token_cfg = None
        self.model_cfg = None
        self.train_cfg = None
        self.data_cfg = None
        self.sched_cfg = None
        self.optim_cfg = None

        # Setup state
        self.use_deepspeed = False
        self.pretrain_cpt = None
        self.output_dir = None
        self.tmp_env = None
        self.world_size = 1
        self.rank = 0

        # EMA
        self.ema_cfg = None
        self.ema_stats = None

        # Set by mode.prepare_data
        self.gtokenizer = None
        self.tokenizer_cls = None
        self.tokenizer_config = None
        self.config = None  # legacy model config

        # Set by _create_model
        self.model = None

        # Set by mode.setup_optimizer
        self.opt_stats = None
        self.device = None

        # Set by mode.setup_training
        self.train_stats = None
        self.tb_writer = None

    def run(self):
        """Execute the full training pipeline."""
        # Phase 0: Shared base setup
        self._extract_config()
        self.mode.update_config(self)
        self._create_ema_config()
        self._setup_deepspeed_flag()
        self._setup_distributed()

        # Phase 1: Data configs (shared)
        self._init_data_configs()

        # Phase 2: Data + tokenizer + sampler + model config (mode-specific)
        self.mode.prepare_data(self)

        # Phase 3: Model creation (shared)
        self._create_model()
        if self.mode.post_model_setup(self):
            return  # early exit (eval_only / infer_only)
        self._load_initial_ckp()

        # Phase 4: Optimizer (mode-specific)
        self.mode.setup_optimizer(self)

        # Phase 5: Resume + save config (shared with mode guards)
        self._resume_checkpoint()
        self._save_model_config()

        # Phase 6: Training preparation (mode-specific)
        self.mode.setup_training(self)

        # Phase 7: Training loop (mode-specific)
        self.mode.run_training(self)

        # Phase 8: Cleanup (shared)
        self._cleanup()

    # ------------------------------------------------------------------ #
    #  Shared private methods
    # ------------------------------------------------------------------ #

    def _extract_config(self):
        """Decompose Hydra config into component sub-configs."""
        self.token_cfg = self.cfg.tokenization
        self.model_cfg = self.cfg.model
        self.train_cfg = self.cfg.training
        self.data_cfg = self.token_cfg.data
        self.sched_cfg = self.train_cfg.schedule
        self.optim_cfg = self.train_cfg.optimizer

    def _create_ema_config(self):
        """Create EMA configuration and stats (config only, no model yet)."""
        self.ema_cfg = EMAConfig(
            use_ema=self.optim_cfg.use_ema,
            ema_file="model_ema.pt",
            ema_file_best="model_ema_best.pt",
        )
        self.ema_stats = EMAStats(ema_cfg=self.ema_cfg)

    def _setup_deepspeed_flag(self):
        """Set DeepSpeed flag and check for resume from existing log."""
        train_cfg = self.train_cfg
        self.pretrain_cpt = train_cfg.pretrain_cpt
        self.output_dir = train_cfg.output_dir

        train_cfg.use_deepspeed = self.use_deepspeed = (
            len(train_cfg.deepspeed_conf_file) > 0
        )

        if os.path.exists(os.path.join(self.output_dir, "log.csv")):
            print(
                f"log file {os.path.join(self.output_dir, 'log.csv')} exists, "
                f"resume training from {self.output_dir} instead of initializing "
                f"from pre-train ckp {self.pretrain_cpt}!"
            )
            self.pretrain_cpt = self.output_dir

    def _setup_distributed(self):
        """Set up distributed training environment."""
        self.tmp_env = misc_utils.set_dist_env(self.train_cfg)
        self.world_size = self.train_cfg.distributed.world_size
        self.rank = self.train_cfg.distributed.rank

    def _init_data_configs(self):
        """Initialize stacked feature, embedding dim, and sync configs."""
        base_configs.init_stacked_feat(self.cfg)
        base_configs.init_embed_dim(self.cfg)
        base_configs.sync_config(self.cfg)

    def _create_model(self):
        """Create model with shared boilerplate: DeepSpeed init, dict_models
        lookup, dict_bounds, gradient checkpointing, cache disable."""
        if self.use_deepspeed:
            deepspeed.init_distributed(
                dist_backend="nccl",
                rank=self.rank,
                world_size=self.world_size,
            )
        self.model = self.mode.dict_models[self.model_cfg.model_type](self.config)
        # For PCQM4M-v2 dataset: propagate dict_bounds to model
        train_ds = getattr(self.mode, '_train_dataset_for_bounds', None)
        if train_ds is not None and hasattr(train_ds, "dict_bounds"):
            self.model.dict_bounds = train_ds.dict_bounds
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False

    def _load_initial_ckp(self):
        """Non-resuming: load from pretrained checkpoint if provided and
        different from current output_dir."""
        self.model = loader_utils.load_from_ckp(
            misc_utils=misc_utils,
            pretrain_cpt=self.pretrain_cpt,
            output_dir=self.output_dir,
            model=self.model,
            config=self.config,
            skip_keys=self.mode.skip_keys,
        )
        print(self.model)

    def _resume_checkpoint(self):
        """Resuming: load model + optimizer from current checkpoint."""
        if not (
            len(self.pretrain_cpt) > 0
            and self.pretrain_cpt == self.output_dir
            and self.mode.allow_resume()
        ):
            return
        ckp, _ = misc_utils.get_latest_ckp(self.pretrain_cpt)
        if self.use_deepspeed:
            print(f"Loading weights from {ckp} with deepspeed API to resume training.")
            self.model.load_checkpoint(ckp)
        else:
            misc_utils.load_ddp_ckp(
                ckp,
                model=self.model,
                optimizer=self.opt_stats.optimizer,
                lr_scheduler=self.opt_stats.lr_scheduler,
            )
        print(
            f"[{datetime.now()}] Finish -> Loading weights from ckp:\n"
            f"{self.model.module.config}"
        )
        self.ema_stats.load_ema_ckp(self.output_dir)

    def _save_model_config(self):
        """Save model config.json to output_dir on rank 0."""
        if self.rank != 0 or not self.mode.allow_save_config():
            return
        try:
            self.model.module.config.save_pretrained(self.output_dir)
        except AttributeError:
            print("In local test setting!!!\n" * 5)
            self.model.config.save_pretrained(self.output_dir)
        print(
            f"[{datetime.now()}] Finish -> Dump model config to "
            f"`{self.output_dir}/config.json`"
        )

    def _cleanup(self):
        """Close TB writer and save final config."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.mode.allow_save_config():
            OmegaConf.save(
                config=self.cfg,
                f=os.path.join(self.output_dir, self.mode.final_config_filename),
            )


def launch(train_fn):
    """Shared __main__ entry point logic for both training scripts.

    Handles multiprocessing setup, sys.argv filtering for DeepSpeed/Hydra
    compatibility, and space-separated argument parsing for Nebula.
    """
    from ..utils import conf_utils

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    print(sys.argv)
    # `--local_rank` is injected by deepspeed, and will not be recognized by Hydra
    sys.argv = [a for a in sys.argv if not a.startswith("--local_rank")]
    assert sys.argv[0].endswith(".py"), f"{sys.argv[0]}"
    if "=" not in sys.argv[-1]:
        print("Parsing space separate arguments in Nebula")
        space_args = sys.argv[1:]
        parsed_config = conf_utils.parse_space_separated_args(space_args)
        hydra_args = []
        for key, value in parsed_config.items():
            if value == "":
                hydra_args.append(f"{key}='{value}'")
            else:
                hydra_args.append(f"{key}={value}")
        sys.argv = [sys.argv[0]] + hydra_args

    train_fn()
