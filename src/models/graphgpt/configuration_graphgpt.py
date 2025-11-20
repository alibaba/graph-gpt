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
        focal_gamma: float = 0,  # param for focal-loss
        smtp_inside: bool = False,  # whether to prepare SMTP inputs and labels inside `model::forward`
        # For downstream tasks
        cls_token_id=None,
        mlp: Optional[List[int]] = None,
        dropout: float = 0,
        loss_type: Optional[str] = None,
        num_neg: Optional[int] = None,
        **kwargs,
    ):
        self.smtp_inside = smtp_inside
        self.focal_gamma = focal_gamma
        self.next_n_token = next_n_token
        self.causal_attention = causal_attention
        self.rope_range = rope_range
        # rope_type: yarn, dynamic, default
        rope_scaling = None  # {"rope_type": "default", "factor": 4}
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
