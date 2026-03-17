# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Shared constants, dataclasses, modules, and init helpers used by both
pretrain and finetune GraphGPT model classes.
"""
import math
from functools import partial

import torch
from dataclasses import dataclass
from torch import nn
from typing import Optional, Tuple

from transformers.utils import ModelOutput
from transformers.models.llama import modeling_llama

from . import utils_graphgpt
from src.utils.attn_mask_utils import _prepare_4d_bi_causal_attention_mask
from src.utils.tokenizer_utils import MOL_ENERGY_BIN_LEN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_prepare_4d_bi_causal_attention_mask = partial(
    _prepare_4d_bi_causal_attention_mask, MOL_ENERGY_BIN_LEN - 1
)  # -1 because the last binary digit won't be used in input_ids, but ONLY in labels
_EPSILON = 1e-7
# POS_TYPE_mask_lookup is only for molecular 3D data
POS_TYPE_mask_lookup = torch.tensor(
    [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.int64
).to(bool)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models with pretrain task and Sequence Classification task.

    Args:
        pretrain_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `pretrain_labels` is provided):
            Language modeling loss.
        task_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `task_labels` is provided):
            E.g, graph-level classification loss.
        pretrain_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        task_logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (`Tuple[Tuple[torch.Tensor]]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of length `config.n_layers`, containing tuples of tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            GPT2Attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    pretrain_loss: Optional[torch.FloatTensor] = None
    task_loss: Optional[torch.FloatTensor] = None
    pretrain_logits: torch.FloatTensor = None
    task_logits: torch.FloatTensor = None
    head1_loss: Optional[torch.FloatTensor] = None
    head2_loss: Optional[torch.FloatTensor] = None
    head1_logits: torch.FloatTensor = None
    head2_logits: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    task_hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# ---------------------------------------------------------------------------
# Shared nn.Module
# ---------------------------------------------------------------------------
class StackedFeatAggregation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.stacked_feat = config.stacked_feat
        if config.stacked_feat_agg_method == "gated":
            self.weight = nn.Parameter(
                torch.empty((config.stacked_feat, config.hidden_size))
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # WHEN loading from pre-trained with HF's `model.from_pretrained`,
        # IF fails due to name mismatch and etc, the params won't be init properly
        # i.e., contains NAN, 0 and exteremely small vals, e.g., 1.4013e-45
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        if self.config.stacked_feat_agg_method == "gated":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        if self.config.stacked_feat_agg_method == "gated":
            x = torch.einsum("nsfd,fd->nsd", x, self.weight)
        else:
            assert (
                self.stacked_feat == x.shape[2]
            ), f"stacked_feat: {self.stacked_feat}\nx.shape: {x.shape}"
            x = torch.sum(x, dim=-2)
        return x

    def extra_repr(self) -> str:
        if self.config.stacked_feat_agg_method == "gated":
            repr_ = f"stacked_feat={self.stacked_feat}, hidden_size={self.hidden_size}"
        else:
            repr_ = "sum(x, dim=-2)"
        return repr_


# ---------------------------------------------------------------------------
# Shared init helpers (free functions taking `self` as first arg)
# ---------------------------------------------------------------------------
def _use_dropout(config):
    if (
        sum([config.path_pdrop, config.mlp_pdrop]) > 0
        or config.layer_scale_init_value > 0
    ):
        print("Applying dropout in backbone transformer")
        return True
    else:
        print("NOT Applying dropout in backbone transformer")
        return False


def init_backbone(self, config):
    """Select and instantiate the LlamaModel backbone (with or without dropout)."""
    LlamaModel = (
        utils_graphgpt.LlamaModel
        if _use_dropout(self.config)
        else modeling_llama.LlamaModel
    )
    if not self.config.causal_attention:
        print("\nSet attention mask to non-causal attention!\n")
    self.model = LlamaModel(config)


def init_embed_dropout(self, config):
    """Set up embedding dropout (or None if embed_pdrop == 0)."""
    if config.embed_pdrop > 0:
        self.embed_dropout = nn.Dropout(p=config.embed_pdrop)
    else:
        self.embed_dropout = None


def init_stacked_feat_agg(self, config, conditional=True):
    """Create StackedFeatAggregation. If conditional=True, only when stack_method is short|long."""
    if conditional and config.stack_method not in {"short", "long"}:
        return
    self.stacked_feat_agg = StackedFeatAggregation(config)


def resolve_forward_defaults(self, output_attentions, output_hidden_states, return_dict, position_ids):
    """Resolve None forward() arguments to config defaults and reset position_ids."""
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )
    position_ids = utils_graphgpt.reset_pos_ids(position_ids, self.config)
    return output_attentions, output_hidden_states, return_dict, position_ids
