from typing import List
from torch import nn
from transformers.activations import ACT2FN

from ..conf import Config


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mlp: List[int],
        hidden_act: str = "linear",
        dropout: float = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.mlp = [in_dim] + list(mlp) + [out_dim]
        ls_modules = []
        for in_dim_, out_dim_ in zip(self.mlp[:-1], self.mlp[1:]):
            module = nn.Linear(in_dim_, out_dim_, bias=bias)
            ls_modules.append(module)
        self.mlp_modules = nn.ModuleList(ls_modules)

        self.act_fn = ACT2FN[hidden_act]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, each_module in enumerate(self.mlp_modules):
            x = self.act_fn(x)
            x = self.dropout(x)
            x = each_module(x)
        return x


def set_up_model_architect(hidden_size: int = 128):
    intermediate_size = hidden_size * 4
    head_dim = 64
    assert hidden_size % head_dim == 0
    num_attention_heads = hidden_size // head_dim
    return intermediate_size, num_attention_heads, head_dim


def freeze_llama_layers(model, freeze_layer_count: int = 0):
    # 1. freeze embeddings
    for name, param in model.model.embed_tokens.named_parameters():
        param.requires_grad = False
        print(f"Freeze param:: model.model.embed_tokens.{name}")
    # 2. freeze transformer layers
    for layer in model.model.layers[:freeze_layer_count]:
        for name, param in layer.named_parameters():
            param.requires_grad = False
            print(f"Freeze param:: model.model.layers.{layer}.{name}")


def set_model_config(cfg: Config, gtokenizer):
    model_config = cfg.model

    task_type = cfg.training.task_type

    # 1. set model architect params
    if (model_config.intermediate_size == 0) and (
        model_config.num_attention_heads == 0
    ):
        (
            model_config.intermediate_size,
            model_config.num_attention_heads,
            model_config.head_dim,
        ) = set_up_model_architect(hidden_size=model_config.hidden_size)
    # 2. set attention type: causal or bi-directional
    model_config.causal_attention = bool(
        0 if task_type == "pretrain-mlm" else model_config.causal_attention
    )
    # 3. set some graph input details
    model_config.pt_head.next_n_token = model_config.graph_input.stacked_feat
    # 4. tokenization related params
    model_config.vocab_size = gtokenizer.vocab_size
    model_config.bos_token_id = gtokenizer.get_bos_token_id()
    model_config.eos_token_id = gtokenizer.get_eos_token_id()
    return model_config


def set_ft_model_config(cfg: Config, gtokenizer):
    # refer to `conf_utils.parse_model_config_for_ft`
    model_config = set_model_config(cfg, gtokenizer)
    if len(cfg.training.pretrain_cpt) == 0:
        model_config.num_key_value_heads = model_config.num_attention_heads
    model_config.tie_word_embeddings = False
    model_config.pt_head.next_n_token = 1
    # TODO: refactor this based on conf_utils.parse_model_config_for_ft
    return model_config
