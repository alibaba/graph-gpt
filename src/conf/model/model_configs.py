from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ===================================================================
# 1. 独立的 Dataclass 子配置
# ===================================================================


@dataclass
class RopeScalingConfig:
    """Configuration for RoPE (Rotary Position Embeddings) scaling."""

    rope_type: str = "default"
    factor: Optional[float] = None
    original_max_position_embeddings: Optional[int] = None
    attention_factor: Optional[float] = None
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    short_factor: List[float] = field(default_factory=list)
    long_factor: List[float] = field(default_factory=list)
    low_freq_factor: Optional[float] = None
    high_freq_factor: Optional[float] = None


@dataclass
class DropoutConfig:
    """Configuration for dropout settings within the model's backbone."""

    embed_dropout: float = 0.0
    path_dropout: float = 0.0
    mlp_dropout: float = 0.0
    attention_dropout: float = 0.0


@dataclass
class GraphInputConfig:
    """Configuration for GraphGPT's specific input feature stacking."""

    stack_method: Optional[str] = None  # short|long
    stacked_feat: int = 1
    stacked_feat_agg_method: str = "sum"
    # Dimension of external embeddings if provided as input of node or edge.
    embed_dim: int = 0


@dataclass
class GeometricInputConfig:
    """Configuration for molecular-specific inputs like 3D positions."""

    pos_agg_method: str = "sum"
    """Aggregation method for 3D positions."""

    pos_bins: int = 512
    """Number of bins for discretizing 3D positions."""


@dataclass
class PretrainingHeadConfig:
    """Configuration for pre-training specific tasks and losses."""

    next_n_token: int = 1
    """Number of next tokens to predict for pre-training."""

    use_generative: bool = True
    """Whether to use generative pre-training."""

    use_discriminative: bool = False
    """Whether to use discriminative pre-training, e.g., Contrastive Learning"""

    focal_gamma: float = 0.0
    """Gamma parameter for focal loss."""

    smtp_inside: bool = False
    """Whether to prepare SMTP inputs and labels inside the model's forward pass."""


@dataclass
class FinetuningHeadConfig:
    """Configuration for downstream fine-tuning tasks."""

    task_type: Optional[str] = None
    """task_type -> pretrain|graph|edge|node; shall be copied from tokenization config"""
    task_ratio: float = 1
    """multi-task loss setting, ratio of sv task"""

    problem_type: Optional[str] = None
    """problem_type -> single_label_classification|multi_label_classification|regression"""

    pooling_method: str = "last"
    """Pooling method for sequence-level representation. One of {'last', 'sum', 'mean'}."""

    mlp: List[int] = field(default_factory=list)
    """A list of hidden sizes for an optional MLP classification head."""

    dropout: float = 0.0
    """Dropout for the final classification head."""

    loss_type: Optional[str] = None
    """Type of loss function for the task (e.g., 'auc', 'cross_entropy')."""

    metric_type: Optional[str] = None

    num_neg: Optional[int] = None
    """Number of negative samples, for losses like 'auc'."""

    num_labels: Optional[int] = None
    """Number of labels for classification tasks; =1 for regression."""


@dataclass
class PositionPretrainingConfig:
    """Config for 3D position prediction and SMTP pretraining objectives (GraphGPTPosPred)."""

    smtp_power: float = 1.0
    """Polynomial scheduler power for SMTP in base pretrain."""

    problem_type: str = "pos-smtp-line"
    """Pretraining problem type: pos-smtp-line|pos-smtp-cube|pos-smtp-mix."""

    smtp_3d_power: float = 1.0
    """3D SMTP scheduler power (polynomial: 0.75/1/1.25, cosine: -1, arccosine: -2)."""

    smtp_3d_noise_scale: float = 0.2
    """Noise scale for 3D SMTP."""

    coord_lvl_mask: bool = True
    """Mask position at per-coordinate level instead of per-position level."""

    num_bins: int = 1024
    """Number of bins for discretizing 3D positions in PosPred."""

    num_bins_line: int = 256
    """Number of bins for line token transformation in PosPred."""

    num_bins_cube: int = 32
    """Number of bins for cube token transformation in PosPred."""

    apply_denoise: bool = False
    """Whether un-masked noisy coords are predicted using clean coords as target."""

    label_smoothing: float = 0.0
    """Label smoothing for cross-entropy loss."""

    pos_agg_method: str = "gated"
    """Aggregation method for 3D position tokens: sum|gated."""

    use_pos_proj: bool = False
    """Whether to use raw-pos input projection."""

    loss_agg: str = "token-lvl"
    """Loss aggregation level: token-lvl|sample-lvl."""

    pos_range: str = "p1p"
    """Position range for discretization."""

    smtp_2d_rate: float = 0.1
    """Rate of samples selected for 2D-SMTP masking."""

    smtp_2d_replace_rate: float = 0.0
    """Rate of replacing [mask] with random tokens (noise injection to 2D tokens)."""

    sep_2d3d_inputs: bool = True
    """Whether 3D/2D-SMTP use separate samples (2D-SMTP samples' pos set to 0)."""

    global_2d_mask: bool = False
    """Whether to apply 2D mask globally on all samples."""

    use_discriminative: bool = False
    """Whether to use discriminative (CL) loss in PosPred."""


@dataclass
class DenoisingRegressionConfig:
    """Config for denoising regression double-heads model."""

    noise_scale: float = 0.35
    """Noise scale for coordinate denoising."""

    denoise_wgt: float = 1.0
    """Weight for denoising loss."""

    denoise_schedule_pow: float = 0.0
    """Power for denoising schedule (0 = no schedule)."""

    bi_causal: bool = False
    """Whether to use bi-directional causal attention."""

    r_2d: float = 4.0
    """Ratio component for 2D mask sampling."""

    r_3d: float = 0.0
    """Ratio component for 3D mask sampling."""

    r_both: float = 6.0
    """Ratio component for 2D&3D combined mask sampling."""

    add_pos_type: bool = True
    """Whether to embed 3D position type."""

    inputs_transform: str = "token-line"
    """Input transform for 3D coordinates: token-line|token-cube|token-mix."""

    num_bins_line: int = 256
    """Number of bins for line token transformation."""

    num_bins_cube: int = 32
    """Number of bins for cube token transformation."""

    pos_range: str = "1p"
    """Position range for discretization."""

    use_pos_proj: bool = False
    """Whether to use raw-pos input projection."""

    smtp_3d: bool = False
    """Whether to add auxiliary 3D-SMTP loss."""

    smtp_wgt: float = 1.0
    """Weight for SMTP loss."""

    smtp_3d_scheduler_power: float = 0.1
    """Scheduler power for 3D-SMTP."""

    smtp_denoise: bool = True
    """Whether un-masked noisy coords are denoised in SMTP."""

    smtp_vocab: int = 256
    """Vocabulary size for SMTP."""

    smtp_2d_rate: float = 0.0
    """Rate of samples for 2D-SMTP masking."""

    smtp_2d_scheduler_power: float = 0.0
    """Scheduler power for 2D-SMTP."""


# ========================================================


# ===================================================================
# 2. 更新主 Dataclass 以使用所有子配置
# ===================================================================


@dataclass
class GraphGPTModelConfig:
    """
    Hydra-compatible merged configuration for GraphGPT with a highly modular structure.
    """

    model_type: str = "graphgpt"
    # -------------------------------------------------------------------
    # 核心模型架构 (Core Model Architecture)
    # -------------------------------------------------------------------
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None

    attention_bias: bool = False
    mlp_bias: bool = False

    hidden_act: str = "gelu"
    max_position_embeddings: int = 1024
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False

    rope_theta: float = 10000.0
    rope_scaling: Optional[RopeScalingConfig] = None
    use_cache: bool = False

    # -------------------------------------------------------------------
    # GraphGPT 特有参数 (General & Graph-Specific Parameters)
    # -------------------------------------------------------------------
    causal_attention: bool = False
    rope_range: int = 0
    layer_scale_init_value: float = 0.0

    # -------------------------------------------------------------------
    # 模块化子配置 (Modular Sub-configs)
    # -------------------------------------------------------------------
    dropout_settings: DropoutConfig = field(default_factory=DropoutConfig)
    graph_input: GraphInputConfig = field(default_factory=GraphInputConfig)
    geometric_input: GeometricInputConfig = field(default_factory=GeometricInputConfig)
    pt_head: PretrainingHeadConfig = field(default_factory=PretrainingHeadConfig)
    pos_pt_head: PositionPretrainingConfig = field(default_factory=PositionPretrainingConfig)
    denoise_head: DenoisingRegressionConfig = field(default_factory=DenoisingRegressionConfig)
    ft_head: FinetuningHeadConfig = field(default_factory=FinetuningHeadConfig)

    # -------------------------------------------------------------------
    # Tokenizer 和元数据 (Tokenizer and Metadata)
    # -------------------------------------------------------------------
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    cls_token_id: Optional[int] = None

    pretraining_tp: int = 1
    keys_to_ignore_at_inference: List[str] = field(
        default_factory=lambda: ["past_key_values"]
    )
    base_model_tp_plan: Dict[str, Any] = field(
        default_factory=lambda: {
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.mlp.gate_proj": "colwise",
            "layers.*.mlp.up_proj": "colwise",
            "layers.*.mlp.down_proj": "rowwise",
        }
    )
    base_model_pp_plan: Dict[str, Any] = field(
        default_factory=lambda: {
            "embed_tokens": (["input_ids"], ["inputs_embeds"]),
            "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
            "norm": (["hidden_states"], ["hidden_states"]),
        }
    )


# ===================================================================
# 3. 示例：如何使用最终的配置
# ===================================================================
if __name__ == "__main__":
    from omegaconf import OmegaConf

    conf = OmegaConf.structured(GraphGPTModelConfig)

    print("--- Fully Modular Configuration ---")
    print(OmegaConf.to_yaml(conf))

    # 访问更深层次的嵌套配置
    print(f"\nDownstream Pooling Method: {conf.ft_head.pooling_method}")
    print(f"3D Position Bins: {conf.molecular_input.pos_bins}")

    # 模拟从命令行覆盖
    cli_overrides = [
        "downstream_task.mlp=[1024, 512]",
        "downstream_task.loss_type=cross_entropy",
        "molecular_input.pos_bins=256",
    ]
    conf_updated = OmegaConf.merge(conf, OmegaConf.from_cli(cli_overrides))

    print("\n--- Updated Downstream Task Config ---")
    print(OmegaConf.to_yaml(conf_updated.downstream_task))
