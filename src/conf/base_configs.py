import math
import os.path
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
from omegaconf import MISSING, OmegaConf
from .tokenization import TokenizationConfig
from .model import GraphGPTModelConfig
from .generation import GenerationConfig

TASK_TYPES = {
    "pretrain",
    "pretrain-smtp",
    "pretrain-mlm",
    "pretrain-mlm-coord",
    "pretrain-coord",
    "pretrain-ltp",
    "pretrain-euler",
    "pretrain-cl",
    "pretrain-coord-cl",
    "node",
    "nodev2",
    "edge",
    "graph",
}


@dataclass
class DistConfig:
    world_size: int = 1
    rank: int = 0


@dataclass
class ScheduleConfig:
    """Configuration for the training schedule, duration, and periodic events."""

    # Special key for Hydra to know which class to instantiate
    # _target_ NOT working, not sure why?
    _target_: str = "src.conf.ScheduleConfig"

    epochs: Optional[int] = None
    warmup_epochs: Optional[float] = None
    total_tokens: float = 1e9
    warmup_tokens: float = 1e8
    total_num_steps: int = MISSING
    warmup_num_steps: int = MISSING
    logging_steps: int = 100
    samples_per_saving: Optional[int] = None
    steps_per_saving: Optional[int] = None
    samples_per_eval: Optional[int] = None  # for finetuning


def update_num_steps(cfg: ScheduleConfig, tokens_per_sample, batch_size, world_size):
    cfg.total_num_steps = int(
        math.ceil(cfg.total_tokens / (tokens_per_sample * batch_size * world_size))
    )
    cfg.warmup_num_steps = int(
        math.ceil(cfg.warmup_tokens / (tokens_per_sample * batch_size * world_size))
    )


def update_epochs(cfg: ScheduleConfig, tokens_per_sample, samples_per_gpu, world_size):
    cfg.epochs = int(
        math.ceil(cfg.total_tokens / (tokens_per_sample * samples_per_gpu * world_size))
    )


def print_stats(cfg: ScheduleConfig):
    print(
        f"\n[{datetime.now()}] total_num_steps: {cfg.total_num_steps}\nwarmup_num_steps: {cfg.warmup_num_steps}\nepochs per worker: {cfg.epochs}\n"
    )


@dataclass
class OptimizerConfig:
    """Dataclass for the optimizer configuration."""

    lr: float = 0.001
    min_lr: float = 0.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.1
    eps: float = 1e-6
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    use_ema: bool = False
    ema_decay: float = 0.9999


@dataclass
class PretrainMlmParams:
    fixed_ratio: float
    power: int
    mtp: List[int]
    umr_clip: List[float]


@dataclass
class PretrainMlmConfig:
    name: str
    params: PretrainMlmParams
    dlm_wgt: bool = True
    num_gen_samples: int = 128  # num of samples picked for generation


@dataclass
class FinetuneTrainConfig:
    """Configuration for finetuning stage."""

    # how to freeze the params of backbone architecture: -1->no, 0->embedding
    freeze: int = -1
    seed: int = -1  # seed to shuffle train sampler
    use_aux: bool = False  # whether to use auxiliary loss
    aux_ratio: float = 0.0  # ratio of auxiliary loss
    task_ratio: float = 1.0  # ratio of task loss


@dataclass
class FinetuneEvalConfig:
    """Configuration for evaluation in finetuning stage."""

    save_pred: bool = False
    save_hidden_states: bool = False
    infer_only: bool = False
    eval_only: bool = False
    epoch_per_eval: int = 1
    k_samplers: int = 262144  # 2^14=16384  2^16=65536  2^18=262144
    true_valid: int = -1


@dataclass
class TrainingConfig:
    deepspeed_conf_file: str = ""
    use_deepspeed: bool = False
    pretrain_cpt: str = ""
    pretrain_mode: bool = False
    gpu_name: str = ""
    task_type: str = "pretrain"  # available vals:: TASK_TYPES
    task_conversion: str = None
    output_dir: str = "../exp/models/graph_llama_test"
    tot_samples: int = 10000  # num of samples for estimating tokens-per-sample
    batch_size: int = 128
    batch_size_eval: Optional[
        int
    ] = None  # small to avoid OOM when evaluating during finetuning
    max_length: Optional[int] = None  # used in finetuning
    pad_to_multiple_of: int = 8
    pack_tokens: float = 0
    num_workers: int = 8  # num of workers for data processing in train DataLoader
    num_workers_eval: int = 8  # num of workers for data processing in eval DataLoader
    valid_percent: float = 0
    do_valid: bool = False
    do_test: bool = False
    do_generation: bool = False
    pt_eval_only: bool = False
    focal_gamma: float = 0  # 0 for CE; other vals for focal-loss
    pretrain_mlm: PretrainMlmConfig = field(default_factory=PretrainMlmConfig)
    distributed: DistConfig = field(default_factory=DistConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    finetune: FinetuneTrainConfig = field(default_factory=FinetuneTrainConfig)
    ft_eval: FinetuneEvalConfig = field(default_factory=FinetuneEvalConfig)


def update_ft_num_steps(train_cfg: TrainingConfig, samples_per_gpu):
    train_cfg.schedule.total_num_steps = train_cfg.schedule.epochs * (
        samples_per_gpu // train_cfg.batch_size
    )
    train_cfg.schedule.warmup_num_steps = int(
        train_cfg.schedule.warmup_epochs * (samples_per_gpu // train_cfg.batch_size)
    )
    print(
        f"\ntotal_num_steps: {train_cfg.schedule.total_num_steps}\nwarmup_num_steps: {train_cfg.schedule.warmup_num_steps}\n"
    )


def set_finetune_cfg(ft_cfg: FinetuneTrainConfig):
    ft_cfg.aux_ratio = 1 - ft_cfg.task_ratio
    ft_cfg.use_aux = ft_cfg.aux_ratio > 0
    print(
        f"ft_cfg:\n{ft_cfg}\naux_ratio: {ft_cfg.aux_ratio}, use_aux: {ft_cfg.use_aux}"
    )


@dataclass
class Config:
    # Main configuration classes
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    model: GraphGPTModelConfig = field(default_factory=GraphGPTModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # Additional parameters not categorized; historical params for experimental ONLY, NOT used ANYMORE
    with_prob: int = 0  # sample graphs for training proportional to their number of eulerian paths/num_nodes
    ignored_off: int = 0
    attr_mask_ratio: int = 0
    logit_adjust: int = 0  # https://spaces.ac.cn/archives/7615
    do_test: int = 0
    eulerian_position: int = 0  # use eulerian position encoding/ntk
    tot_samples: int = 10000  # num of samples for estimating tokens-per-sample
    rope_scaling_factor: int = 0  # finetuning
    cyclic_mpe: int = 0  # whether to use cycle mpe; finetuning


def init_stacked_feat(cfg: Config):
    model_cfg = cfg.model
    token_cfg = cfg.tokenization

    if token_cfg.tokenizer_class == "StackedGSTTokenizer":
        attr_dim = token_cfg.semantics.edge.dim + token_cfg.semantics.node.dim
        assert model_cfg.graph_input.stack_method in (
            "short",
            "long",
            None,
        ), f"stack_method: {model_cfg.graph_input.stack_method}"
        if token_cfg.structure.edge.remove_edge_type_token:
            stacked_feat = 1 + attr_dim
        else:
            stacked_feat = 2 + attr_dim
    else:
        stacked_feat = 1
    model_cfg.graph_input.stacked_feat = stacked_feat


def init_embed_dim(cfg: Config):
    model_cfg = cfg.model
    token_cfg = cfg.tokenization

    if "pcqm4m" in token_cfg.data.data_path:
        if model_cfg.model_type == "graphgpt-denoise":
            token_cfg.semantics.node.embed = "pos"
            token_cfg.semantics.node.embed_dim = 3

    model_cfg.graph_input.embed_dim = (token_cfg.semantics.node.embed_dim or 0) + (
        token_cfg.semantics.edge.embed_dim or 0
    )


def sync_config(cfg: Config):
    model_cfg = cfg.model
    train_cfg = cfg.training

    model_cfg.ft_head.task_type = train_cfg.task_type
    train_cfg.max_length = train_cfg.max_length or model_cfg.max_position_embeddings


def update_cfg_with_saved_cfg_yaml(cfg: Config):
    if cfg.training.ft_eval.eval_only:
        save_hidden_states = cfg.training.ft_eval.save_hidden_states
        fn = os.path.join(cfg.training.output_dir, "config_final.yaml")
        if not os.path.exists(fn):
            fn = os.path.join(cfg.training.output_dir, "config.yaml")
        extra_cfg = OmegaConf.load(fn)
        cfg = OmegaConf.merge(cfg, extra_cfg)
        cfg.training.ft_eval.eval_only = True
        cfg.training.ft_eval.save_pred = False
        cfg.training.ft_eval.save_hidden_states = save_hidden_states
        print(
            f"In [EVAL-ONLY] mode, merging cfg with config from yaml:\n{fn}\n{OmegaConf.to_yaml(cfg)}"
        )
    return cfg


def update_odps_cfg_from_token_cfg(cfg: Config, mode: str):
    token_cfg = cfg.tokenization
    odps_cfg = token_cfg.data.odps
    if odps_cfg.tables != "":
        odps_cfg.edge_dim = token_cfg.semantics.edge.dim
        odps_cfg.node_dim = token_cfg.semantics.node.dim
        odps_cfg.mode = mode


def update_odps_cfg_for_ft_infer(cfg: Config):
    if cfg.training.ft_eval.infer_only:
        cfg.training.ft_eval.eval_only = True
        tables = cfg.tokenization.data.odps.tables
        outputs = cfg.tokenization.data.odps.outputs
        assert (
            len(tables.split(",")) == 1
        ), f"len(tables.split(',')): {len(tables.split(','))}"
        assert (
            len(outputs.split(",")) == 1
        ), f"len(outputs.split(',')): {len(outputs.split(','))}"
        cfg.tokenization.data.odps.tables = ",".join([tables] * 3)


def update_finetune_cfg(cfg: Config):
    if cfg.training.finetune.seed < 0:
        cfg.training.finetune.seed = int(datetime.now().date().strftime("%Y%m%d")[::-1])


def update_generation_cfg(cfg: Config, gtokenizer):
    gen_cfg = cfg.generation
    gen_cfg.mask_token_id = gtokenizer.get_mask_token_id()
    gen_cfg.pad_token_id = gtokenizer.pad_token_id
    gen_cfg.bos_token_id = gtokenizer.get_bos_token_id()
    gen_cfg.eos_token_id = gtokenizer.get_eos_token_id()
    gen_cfg.temperature = 1
