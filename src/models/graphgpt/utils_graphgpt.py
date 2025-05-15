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
Implement llama with dropout by referring to transformers/models/llama/modeling_llama.py
and transformers/models/gpt2/modeling_gpt2.py

Dropout to be implemented in 3 modules
1. LlamaMLP -> GPT2MLP :: mlp dropout
2. LlamaAttention -> GPT2Attention :: attention dropout
3. LlamaModel -> GPT2Model :: token embedding dropout
"""
import warnings
from typing import List, Optional, Tuple, Union, Callable
import numpy as np
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from transformers.utils import logging
from transformers.models.llama import modeling_llama
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.beit.modeling_beit import BeitDropPath
from transformers.utils.import_utils import is_torch_fx_available
from src.utils.attn_mask_utils import is_torch_greater_or_equal_than_1_13
from src.utils.attn_mask_utils import (
    _prepare_4d_causal_bi_attention_mask,
    _prepare_4d_attention_mask,
)

apply_rotary_pos_emb = modeling_llama.apply_rotary_pos_emb
repeat_kv = modeling_llama.repeat_kv

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_attention_mask = torch.fx.wrap(_prepare_4d_attention_mask)

logger = logging.get_logger(__name__)


class DropPath(BeitDropPath):
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__(drop_prob)


class LlamaMLP(modeling_llama.LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.mlp_act_dropout = nn.Dropout(config.mlp_pdrop)
        self.mlp_dropout = nn.Dropout(config.mlp_pdrop)

    def forward(self, x):
        assert not (self.config.pretraining_tp > 1)
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        x = self.mlp_act_dropout(x)
        down_proj = self.mlp_dropout(self.down_proj(x))
        return down_proj


class LlamaDecoderLayer(modeling_llama.LlamaDecoderLayer):
    def __init__(
        self, config: LlamaConfig, layer_idx: int, drop_prob: Optional[float] = None
    ):
        super().__init__(config, layer_idx)
        if config.mlp_pdrop > 0:
            del self.mlp
            self.mlp = LlamaMLP(config)
        if drop_prob is None:
            drop_prob = config.path_pdrop
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()
        # copied from transformers.models.beit.modeling_beit.BeitLayer
        init_values = config.layer_scale_init_value
        if init_values > 0:
            self.lambda_1 = nn.Parameter(
                init_values * torch.ones(config.hidden_size), requires_grad=True
            )
            self.lambda_2 = nn.Parameter(
                init_values * torch.ones(config.hidden_size), requires_grad=True
            )
        else:
            self.lambda_1, self.lambda_2 = None, None
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        # apply lambda_1 if present TODO: indicator of modification
        if self.lambda_1 is not None:
            hidden_states = self.lambda_1 * hidden_states
        # first residual connection
        hidden_states = residual + self.drop_path(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # apply lambda_2 if present  TODO: indicator of modification
        if self.lambda_2 is not None:
            hidden_states = self.lambda_2 * hidden_states
        # second residual connection
        hidden_states = residual + self.drop_path(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaModel(modeling_llama.LlamaModel):
    """
    Code modified from https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/modeling_llama.py
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # stochastic depth decay rule
        dpr = np.linspace(0, config.path_pdrop, config.num_hidden_layers)
        del self.layers
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx, dpr[layer_idx])
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # Initialize weights and apply final processing
        self.post_init()


def get_delta_pos(pos):
    # refer to: https://github.com/lsj2408/Transformer-M/blob/45647e143a5282e0e97117969396446084bcf1ab/Transformer-M/modules/transformer_m_layers.py#L179
    # => (r_i - r_j)/||r_i - r_j||
    n_graph, n_node, _ = pos.shape  # [bsz, seq, 3]
    # [bsz, 1, seq, 3] & [bsz, seq, 1, 3] -> [bsz, seq, seq, 3]
    delta_pos_raw = pos.unsqueeze(1) - pos.unsqueeze(2)
    # [bsz, seq, seq]
    dist = delta_pos_raw.norm(dim=-1).view(-1, n_node, n_node)
    # [bsz, seq, seq, 3]
    delta_pos_norm = delta_pos_raw / (dist.unsqueeze(-1) + 1e-5)
    return delta_pos_norm, delta_pos_raw


def apply_sample_lvl_mask_pos(raw_pos, mask_pos_ratio):
    # raw_pos: [bz, seq, 3], mask_pos_ratio: scalar
    bsz = raw_pos.size()[0]
    device = raw_pos.device
    sample_mask = torch.rand((bsz, 1, 1), device=device) < mask_pos_ratio
    return raw_pos.masked_fill_(sample_mask, 0.0)  # [bsz, seq, 3]


def apply_sample_lvl_mask_alternative(raw_pos):
    # raw_pos: [bz, seq, 3]
    bsz = raw_pos.size(0)
    device = raw_pos.device
    sample_mask = (torch.arange(bsz, device=device) % 2).to(bool)[:, None, None]
    return raw_pos.masked_fill_(sample_mask, 0.0)  # [bsz, seq, 3]


def mask_2d(input_ids, noise_mask, mask_2d_ratio):
    # input_ids: [bz, seq, num_feat], noise_mask: [bz, seq, 1], mask_2d_ratio: float
    # print(f"[DEBUG] invoking `mask_2d`\ninput_ids:\n{input_ids}\nnoise_mask:\n{noise_mask}\nmask_2d_ratio: {mask_2d_ratio}")
    bz, seq, num_feat = input_ids.shape
    device = input_ids.device
    mask_2d_token_id = 0  # 1|0 -> which better?
    sample_mask_3d = noise_mask[:, :, 0].all(dim=-1)  # [bz]
    # print(f"[DEBUG] sample_mask_3d:\n{sample_mask_3d}")
    sample_mask_2d = torch.rand((bz,), device=device) < mask_2d_ratio
    # print(f"[DEBUG] sample_mask_2d:\n{sample_mask_2d}")
    sample_mask = (~sample_mask_3d) & sample_mask_2d  # [bz]
    # print(f"[DEBUG] sample_mask:\n{sample_mask}")
    sample_mask = sample_mask[:, None, None].expand(bz, seq, num_feat - 1).contiguous()
    # below ensure the input_ids with 0-pad won't be replaced by `mask_2d_token_id`
    sample_mask = sample_mask & (input_ids[:, :, 0:1] > 0)
    # print(f"[DEBUG] sample_mask:\n{sample_mask}")
    unmask_node_idx = torch.zeros((bz, seq, 1), dtype=bool, device=device)
    sample_mask = torch.cat([unmask_node_idx, sample_mask], dim=-1)
    # print(f"[DEBUG] sample_mask:\n{sample_mask}")
    input_ids = input_ids.masked_fill_(sample_mask, mask_2d_token_id)
    # print(f"[DEBUG] input_ids:\n{input_ids}")
    return input_ids


def get_denoise_loss(logits, noise_mask, noise):
    # logits: [bsz, seq, dim], noise_mask: [bsz, seq, 1], noise: [bsz, seq, 3]
    # refer to: https://github.com/lsj2408/Transformer-M/blob/45647e143a5282e0e97117969396446084bcf1ab/Transformer-M/criterions/graph_prediction.py#L58
    # different from above, our noise_mask already covers `mask` on 3d pos samples
    # see: https://github.com/lsj2408/Transformer-M/blob/45647e143a5282e0e97117969396446084bcf1ab/Transformer-M/modules/transformer_m_encoder.py#L292
    logits = logits.masked_fill_(noise_mask, 0.0)  # [bsz, seq, 3]
    # [bsz, seq, 3] & [bsz, seq, 3] -> [bsz, seq]
    node_output_loss = nn.CosineSimilarity(dim=-1)(logits.float(), noise.float())
    node_output_loss = 1 - node_output_loss
    # [bsz, seq] -> [bsz]
    node_output_loss = (
        node_output_loss.masked_fill_(noise_mask.squeeze(-1), 0.0)
        .sum(dim=-1)
        .to(logits.dtype)
    )
    # [bsz, seq, 1] -> [bsz, seq] -> [bsz]
    tgt_count = (~noise_mask).squeeze(-1).sum(dim=-1).to(node_output_loss)
    tgt_count = tgt_count.masked_fill_(tgt_count == 0.0, 1.0)  # [bsz]
    node_output_loss = (node_output_loss / tgt_count).mean()
    return node_output_loss


class AtomTaskHead(modeling_llama.LlamaAttention):
    """
    refer to: https://github.com/lsj2408/Transformer-M/blob/45647e143a5282e0e97117969396446084bcf1ab/Transformer-M/modules/transformer_m_layers.py#L247
    mixed above with `modeling_llama.LlamaAttention`
    """

    def __init__(self, config):
        super().__init__(config)
        self.is_causal = False
        self.embed_dim = self.hidden_size
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(self.embed_dim, 1)
        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(self.embed_dim, 1)
        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(self.embed_dim, 1)
        del self.o_proj
        self.scaling = (self.embed_dim // self.num_heads) ** -0.5
        self.dropout_module = nn.Dropout(p=self.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        delta_pos: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """
        :param hidden_states: [bsz, seq, dim]
        :param position_ids:
        :param delta_pos: [bsz, seq, seq, 3]
        :param kwargs:
        :return:
        """
        # In Transformer-M, input `query` is [seq, bsz, dim]
        # In GraphGPT, input `query` is [bsz, seq, dim]
        if position_ids is None:
            position_ids = torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)
        # BELOW merged with `modeling_llama.LlamaSdpaAttention`
        query = hidden_states
        bsz, n_node, _ = query.size()
        # [bsz, seq, heads, dim] -> [bsz, heads, seq, dim]
        q = self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        q = q * self.scaling
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = modeling_llama.apply_rotary_pos_emb(q, k, cos, sin)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        attn_probs_float = torch.softmax(attn.view(-1, n_node, n_node), dim=-1)
        attn_probs = attn_probs_float.type_as(attn)  # [bsz*head, n, n]
        attn_probs = self.dropout_module(attn_probs).view(
            bsz, self.num_heads, n_node, n_node
        )
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 1] & [bsz, 1, n, n, 3] -> [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        # [bsz, head, 3, n, n] & [bsz, head, 1, n, dim] -> [bsz, head, 3, n, d]
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head, 3, n, d]
        # -> [bsz, n, 3, head, d] -> [bsz, n, 3, head*d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force


class FocalLoss(nn.Module):
    """
    adapted from: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """

    def __init__(self, gamma=0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        assert reduction in {"none", "mean", "sum"}, f"reduction: {reduction}"
        self.reduction = reduction

    def forward(self, logits, target):
        assert logits.dim() <= 2, f"logits.dim(): {logits.dim()} > 2"
        target = target.view(-1, 1)

        logpt = F.log_softmax(logits, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != logits.data.type():
                self.alpha = self.alpha.type_as(logits.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss


class RotaryEmbedding3D(nn.Module):
    """
    https://spaces.ac.cn/archives/10352
    https://spaces.ac.cn/archives/8265
    """

    def __init__(self, dim, base=10000, device=None):
        super().__init__()
        start = -dim // 2
        # rename `inv_freq` to be `freq`, because it's actually frequency
        freq = 1.0 / (
            base
            ** (
                torch.arange(start, start + dim, 2, dtype=torch.int64)
                .float()
                .to(device)
                / dim
            )
        )
        # freq -> [d/2]
        # if base==10000, freq max==100, min==0.01
        self.dim = dim
        self.base = base
        self.start = start
        self.register_buffer("freq", freq, persistent=False)
        self.expand_rate = int(np.ceil((dim // 2) / 3.0))  # 3 -> 3d

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        """
        :param x: [bs, num_attention_heads, seq_len, head_size]
        :param position_ids: [bs, seq_len, 3]; In `LlamaRotaryEmbedding`, [1, seq_len]
        :param seq_len:
        :return:
        ```python
        bs = position_ids.shape[0]
        # [d/2] -> [bs, d/2, 1]
        freq_expanded = self.freq[None, :, None].float().expand(bs, -1, 1)
        # In `LlamaRotaryEmbedding`:: [1, seq_len] -> [1, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # In `LlamaRotaryEmbedding`:: [bs, d/2, 1] & [1, 1, seq_len] -> [bs, d/2, seq_len] -> [bs, seq_len, d/2]
            freqs = (freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # [bs, seq_len, d/2] -> [bs, seq_len, d]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        ```
        """
        if seq_len is not None:
            logger.warning_once(
                "The `seq_len` argument is deprecated and unused. It will be removed in v4.39."
            )

        bs, seq, _ = position_ids.shape
        # [d/2] -> [1, 1, d/2]
        freq_expanded = self.freq[None, None, :].float()
        # [bs, seq_len, 3] -> [bs, seq_len, 3, 1] -> [bs, seq_len, 3, expand_rate]
        position_ids_expanded = (
            position_ids[:, :, :, None]
            .expand(-1, -1, -1, self.expand_rate)
            .contiguous()
        )
        # [bs, seq_len, 3, expand_rate] -> [bs, seq_len, expand_rate, 3] -> [bs, seq_len, expand_rate*3] -> [bs, seq_len, d/2]
        position_ids_expanded = (
            position_ids_expanded.transpose(2, 3)
            .reshape(bs, seq, -1)[:, :, : self.dim // 2]
            .float()
            .contiguous()
        )
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # [bs, seq_len, d/2] & [1, 1, d/2] -> [bs, seq_len, d/2]
            freqs = position_ids_expanded.float() * freq_expanded.float()
            # [bs, seq_len, d/2] -> [bs, seq_len, d]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def get_step_pos_emb(dim, mpe):
    # 1. get freqs
    periods = torch.arange(0, dim // 2, 1, dtype=torch.int64) + 1
    freqs = 2 * torch.pi / periods.float()
    # 2. step_pos @
    step_pos = torch.arange(0, mpe, 1, dtype=torch.int64)
    step_pos_emb = torch.matmul(
        step_pos.reshape((-1, 1)).float(), freqs.reshape((1, -1))
    )  # [mpe, d/2]
    step_pos_emb_cos = step_pos_emb.cos()
    step_pos_emb_sin = step_pos_emb.sin()
    step_pos_emb_cat = torch.cat(
        [step_pos_emb_cos, step_pos_emb_sin], dim=-1
    )  # [mpe, dim]
    # 3. re-order embed matrix
    idx_slice = [[x, dim // 2 + x] for x in range(dim // 2)]
    idx_slice = [x for ele in idx_slice for x in ele]
    step_pos_emb_all = step_pos_emb_cat[:, idx_slice].clone()
    return step_pos_emb_all


def reset_pos_ids(position_ids: torch.Tensor, config):
    if (config.rope_range > 0) and (position_ids is not None):
        position_ids = (
            position_ids.float()
            * config.rope_range
            / (position_ids.max(dim=-1, keepdim=True).values + 1).float()
        )
    return position_ids
