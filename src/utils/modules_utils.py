import torch
import torch.nn.functional as F
from typing import List
from torch import nn
from transformers.activations import ACT2FN


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


def set_up_model_architect(hidden_size: int = 128, num_hidden_layers: int = 2):
    intermediate_size = hidden_size * 4
    assert hidden_size % 64 == 0
    num_attention_heads = hidden_size // 64
    return hidden_size, intermediate_size, num_attention_heads, num_hidden_layers


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
