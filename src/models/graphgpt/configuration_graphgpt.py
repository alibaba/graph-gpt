from typing import Optional, List, Any
from transformers import LlamaConfig
from ...conf.model import GraphGPTModelConfig


class GraphGPTConfig(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`GraphLlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`LlamaConfig`] and can be used to control the model outputs. Read the
    documentation from [`LlamaConfig`] for more information.


    Args:

        pad_token_id (`int`, *optional*, defaults to 0):
        cls_token_id (`int`, *optional*, defaults to None):
        num_neg (`int`, *optional*, defaults to None): For auc loss ONLY
        Example:
    """

    model_type = "graphgpt"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        tie_word_embeddings=False,
        pooling_method="last",
        causal_attention: bool = True,
        rope_range: int = 0,
        # For transformer backbone's dropout -> embed layer, attn & attn residue, mlp layer
        embed_pdrop: float = 0,
        path_pdrop: float = 0,
        mlp_pdrop: float = 0,
        layer_scale_init_value: float = 0,
        # For graphGPT input features stack
        stacked_feat: int = 1,
        stack_method: str = None,  # short|long
        stacked_feat_agg_method: str = "sum",
        pos_agg_method: str = "sum",  # for mol's 3D positions
        pos_bins: int = 512,  # for mol's 3D positions
        # For input with embed features
        embed_dim: int = 0,
        # For pre-train task
        next_n_token: int = 1,
        use_generative: bool = True,
        use_discriminative: bool = False,
        focal_gamma: float = 0,  # param for focal-loss
        smtp_inside: bool = False,  # whether to prepare SMTP inputs and labels inside `model::forward`
        # For downstream tasks
        cls_token_id=None,
        mlp: Optional[List[int]] = None,
        dropout: float = 0,
        loss_type: Optional[str] = None,
        num_neg: Optional[int] = None,
        # --- Position pretraining head (GraphGPTPosPred) ---
        smtp_power: float = 1.0,
        pt_problem_type: str = "pos-smtp-line",
        smtp_3d_power: float = 1.0,
        smtp_3d_noise_scale: float = 0.2,
        coord_lvl_mask: bool = True,
        pt_num_bins: int = 1024,
        pt_num_bins_line: int = 256,
        pt_num_bins_cube: int = 32,
        apply_denoise: bool = False,
        label_smoothing: float = 0.0,
        pt_pos_agg_method: str = "gated",
        use_pos_proj: bool = False,
        loss_agg: str = "token-lvl",
        pt_pos_range: str = "p1p",
        pt_smtp_2d_rate: float = 0.1,
        smtp_2d_replace_rate: float = 0.0,
        sep_2d3d_inputs: bool = True,
        global_2d_mask: bool = False,
        pt_use_discriminative: bool = False,
        # --- Denoising regression head ---
        noise_scale: float = 0.35,
        denoise_wgt: float = 1.0,
        denoise_schedule_pow: float = 0.0,
        bi_causal: bool = False,
        r_2d: float = 4.0,
        r_3d: float = 0.0,
        r_both: float = 6.0,
        add_pos_type: bool = True,
        inputs_transform: str = "token-line",
        num_bins_line: int = 256,
        num_bins_cube: int = 32,
        dn_pos_range: str = "1p",
        dn_use_pos_proj: bool = False,
        smtp_3d: bool = False,
        smtp_wgt: float = 1.0,
        smtp_3d_scheduler_power: float = 0.1,
        smtp_denoise: bool = True,
        smtp_vocab: int = 256,
        dn_smtp_2d_rate: float = 0.0,
        smtp_2d_scheduler_power: float = 0.0,
        **kwargs,
    ):
        self.causal_attention = causal_attention
        self.rope_range = rope_range
        # rope_type: yarn, dynamic, default
        rope_scaling = None  # {"rope_type": "default", "factor": 4}
        # pre-train task settings
        self.next_n_token = next_n_token
        self.use_generative = use_generative
        self.use_discriminative = use_discriminative
        self.focal_gamma = focal_gamma
        self.smtp_inside = smtp_inside
        # 1. For dropout in transformer backbone
        self.embed_pdrop = embed_pdrop
        self.path_pdrop = path_pdrop
        self.mlp_pdrop = mlp_pdrop
        self.layer_scale_init_value = layer_scale_init_value
        self.stacked_feat = stacked_feat
        self.stack_method = stack_method
        self.stacked_feat_agg_method = stacked_feat_agg_method
        self.pos_agg_method = pos_agg_method
        self.pos_bins = pos_bins
        self.embed_dim = embed_dim
        # 2. For downstream tasks
        self.cls_token_id = cls_token_id
        self.mlp = [] if mlp is None else list(mlp)
        assert pooling_method in {"last", "sum", "mean"}
        self.pooling_method = pooling_method
        self.dropout = dropout
        self.loss_type = loss_type
        self.num_neg = num_neg
        self.rope_3d = False
        # 3. Position pretraining head (GraphGPTPosPred)
        self.smtp_power = smtp_power
        self.pt_problem_type = pt_problem_type
        self.smtp_3d_power = smtp_3d_power
        self.smtp_3d_noise_scale = smtp_3d_noise_scale
        self.coord_lvl_mask = coord_lvl_mask
        self.pt_num_bins = pt_num_bins
        self.pt_num_bins_line = pt_num_bins_line
        self.pt_num_bins_cube = pt_num_bins_cube
        self.apply_denoise = apply_denoise
        self.label_smoothing = label_smoothing
        self.pt_pos_agg_method = pt_pos_agg_method
        self.use_pos_proj = use_pos_proj
        self.loss_agg = loss_agg
        self.pt_pos_range = pt_pos_range
        self.pt_smtp_2d_rate = pt_smtp_2d_rate
        self.smtp_2d_replace_rate = smtp_2d_replace_rate
        self.sep_2d3d_inputs = sep_2d3d_inputs
        self.global_2d_mask = global_2d_mask
        self.pt_use_discriminative = pt_use_discriminative
        # 4. Denoising regression head
        self.noise_scale = noise_scale
        self.denoise_wgt = denoise_wgt
        self.denoise_schedule_pow = denoise_schedule_pow
        self.bi_causal = bi_causal
        self.r_2d = r_2d
        self.r_3d = r_3d
        self.r_both = r_both
        self.add_pos_type = add_pos_type
        self.inputs_transform = inputs_transform
        self.num_bins_line = num_bins_line
        self.num_bins_cube = num_bins_cube
        self.dn_pos_range = dn_pos_range
        self.dn_use_pos_proj = dn_use_pos_proj
        self.smtp_3d = smtp_3d
        self.smtp_wgt = smtp_wgt
        self.smtp_3d_scheduler_power = smtp_3d_scheduler_power
        self.smtp_denoise = smtp_denoise
        self.smtp_vocab = smtp_vocab
        self.dn_smtp_2d_rate = dn_smtp_2d_rate
        self.smtp_2d_scheduler_power = smtp_2d_scheduler_power
        print(
            f"[BEFORE] hidden_size: {hidden_size}, num_attention_heads: {num_attention_heads}"
        )
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_scaling=rope_scaling,
            **kwargs,
        )
        print(
            f"[AFTER] head_dim: {self.head_dim}, hidden_size: {self.hidden_size}, num_attention_heads: {self.num_attention_heads}"
        )

    def update(self, config_dict: dict[str, Any]):
        super().update(config_dict)
        print(f"updated: {config_dict}\n")


def convert_to_legacy_config(model_config: GraphGPTModelConfig) -> GraphGPTConfig:
    """
    Converts a structured GraphGPTModelConfig instance into a legacy, flat
    GraphGPTConfig instance for backward compatibility.

    Args:
        model_config: An instance of the new `GraphGPTModelConfig`.

    Returns:
        An instance of the old `GraphGPTConfig`.
    """
    # Create a flat dictionary of keyword arguments by extracting values
    # from the structured model_config.
    kwargs = {
        # Core Llama/Transformer parameters
        "vocab_size": model_config.vocab_size,
        "hidden_size": model_config.hidden_size,
        "intermediate_size": model_config.intermediate_size,
        "num_hidden_layers": model_config.num_hidden_layers,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": model_config.num_key_value_heads,
        "head_dim": model_config.head_dim,
        "attention_bias": model_config.attention_bias,
        "mlp_bias": model_config.mlp_bias,
        "hidden_act": model_config.hidden_act,
        "max_position_embeddings": model_config.max_position_embeddings,
        "initializer_range": model_config.initializer_range,
        "rms_norm_eps": model_config.rms_norm_eps,
        "tie_word_embeddings": model_config.tie_word_embeddings,
        "rope_theta": model_config.rope_theta,
        "use_cache": model_config.use_cache,
        # Tokenizer 相关
        "pad_token_id": model_config.pad_token_id,
        "bos_token_id": model_config.bos_token_id,
        "eos_token_id": model_config.eos_token_id,
        "cls_token_id": model_config.cls_token_id,
        # Top-level GraphGPT-specific parameters
        "causal_attention": model_config.causal_attention,
        "rope_range": model_config.rope_range,
        "layer_scale_init_value": model_config.layer_scale_init_value,
        # Parameters from `dropout_settings`
        "embed_pdrop": model_config.dropout_settings.embed_dropout,
        "path_pdrop": model_config.dropout_settings.path_dropout,
        "mlp_pdrop": model_config.dropout_settings.mlp_dropout,
        "attention_dropout": model_config.dropout_settings.attention_dropout,
        # Parameters from `graph_input`
        "stacked_feat": model_config.graph_input.stacked_feat,
        "stack_method": model_config.graph_input.stack_method,
        "stacked_feat_agg_method": model_config.graph_input.stacked_feat_agg_method,
        "embed_dim": model_config.graph_input.embed_dim,
        # Parameters from `geometric_input`
        "pos_agg_method": model_config.geometric_input.pos_agg_method,
        "pos_bins": model_config.geometric_input.pos_bins,
        # Parameters from `pretraining head`
        "next_n_token": model_config.pt_head.next_n_token,
        "use_generative": model_config.pt_head.use_generative,
        "use_discriminative": model_config.pt_head.use_discriminative,
        "focal_gamma": model_config.pt_head.focal_gamma,
        "smtp_inside": model_config.pt_head.smtp_inside,
        # Parameters from `finetuning head`
        "pooling_method": model_config.ft_head.pooling_method,
        "mlp": model_config.ft_head.mlp,
        "dropout": model_config.ft_head.dropout,
        "loss_type": model_config.ft_head.loss_type,
        "num_neg": model_config.ft_head.num_neg,
        "num_labels": model_config.ft_head.num_labels,
        "problem_type": model_config.ft_head.problem_type,
        "use_aux": model_config.ft_head.task_ratio < 1,
        # Parameters from `pos_pt_head` (position pretraining)
        "smtp_power": model_config.pos_pt_head.smtp_power,
        "pt_problem_type": model_config.pos_pt_head.problem_type,
        "smtp_3d_power": model_config.pos_pt_head.smtp_3d_power,
        "smtp_3d_noise_scale": model_config.pos_pt_head.smtp_3d_noise_scale,
        "coord_lvl_mask": model_config.pos_pt_head.coord_lvl_mask,
        "pt_num_bins": model_config.pos_pt_head.num_bins,
        "pt_num_bins_line": model_config.pos_pt_head.num_bins_line,
        "pt_num_bins_cube": model_config.pos_pt_head.num_bins_cube,
        "apply_denoise": model_config.pos_pt_head.apply_denoise,
        "label_smoothing": model_config.pos_pt_head.label_smoothing,
        "pt_pos_agg_method": model_config.pos_pt_head.pos_agg_method,
        "use_pos_proj": model_config.pos_pt_head.use_pos_proj,
        "loss_agg": model_config.pos_pt_head.loss_agg,
        "pt_pos_range": model_config.pos_pt_head.pos_range,
        "pt_smtp_2d_rate": model_config.pos_pt_head.smtp_2d_rate,
        "smtp_2d_replace_rate": model_config.pos_pt_head.smtp_2d_replace_rate,
        "sep_2d3d_inputs": model_config.pos_pt_head.sep_2d3d_inputs,
        "global_2d_mask": model_config.pos_pt_head.global_2d_mask,
        "pt_use_discriminative": model_config.pos_pt_head.use_discriminative,
        # Parameters from `denoise_head` (denoising regression)
        "noise_scale": model_config.denoise_head.noise_scale,
        "denoise_wgt": model_config.denoise_head.denoise_wgt,
        "denoise_schedule_pow": model_config.denoise_head.denoise_schedule_pow,
        "bi_causal": model_config.denoise_head.bi_causal,
        "r_2d": model_config.denoise_head.r_2d,
        "r_3d": model_config.denoise_head.r_3d,
        "r_both": model_config.denoise_head.r_both,
        "add_pos_type": model_config.denoise_head.add_pos_type,
        "inputs_transform": model_config.denoise_head.inputs_transform,
        "num_bins_line": model_config.denoise_head.num_bins_line,
        "num_bins_cube": model_config.denoise_head.num_bins_cube,
        "dn_pos_range": model_config.denoise_head.pos_range,
        "dn_use_pos_proj": model_config.denoise_head.use_pos_proj,
        "smtp_3d": model_config.denoise_head.smtp_3d,
        "smtp_wgt": model_config.denoise_head.smtp_wgt,
        "smtp_3d_scheduler_power": model_config.denoise_head.smtp_3d_scheduler_power,
        "smtp_denoise": model_config.denoise_head.smtp_denoise,
        "smtp_vocab": model_config.denoise_head.smtp_vocab,
        "dn_smtp_2d_rate": model_config.denoise_head.smtp_2d_rate,
        "smtp_2d_scheduler_power": model_config.denoise_head.smtp_2d_scheduler_power,
    }

    # 处理 rope_scaling 配置
    if model_config.rope_scaling:
        kwargs["rope_scaling"] = {
            "rope_type": model_config.rope_scaling.rope_type,
            "factor": model_config.rope_scaling.factor,
            "original_max_position_embeddings": model_config.rope_scaling.original_max_position_embeddings,
            "attention_factor": model_config.rope_scaling.attention_factor,
            "beta_fast": model_config.rope_scaling.beta_fast,
            "beta_slow": model_config.rope_scaling.beta_slow,
            "short_factor": model_config.rope_scaling.short_factor,
            "long_factor": model_config.rope_scaling.long_factor,
            "low_freq_factor": model_config.rope_scaling.low_freq_factor,
            "high_freq_factor": model_config.rope_scaling.high_freq_factor,
        }

    # 移除值为 None 的参数，避免 GraphGPTConfig 的验证错误
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # Instantiate the legacy config class with the flattened arguments
    legacy_config = GraphGPTConfig(**kwargs)

    return legacy_config
