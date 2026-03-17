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
Standalone helper functions used by GraphGPT model classes.
Organized by functional category.
"""
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional

from . import utils_graphgpt
from .modeling_common import _EPSILON, _prepare_4d_bi_causal_attention_mask
from src.utils.loss_utils import _dist_infonce
from src.utils.mol_utils import discrete_pos
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


# ===========================================================================
# A. Attention mask utilities
# ===========================================================================
def _update_causal_mask(self, attention_mask, input_tensor, **kwargs):
    if hasattr(self, "bi_causal") and self.bi_causal:
        return _prepare_4d_bi_causal_attention_mask(attention_mask, input_tensor.dtype)
    if len(attention_mask.size()) == 2:
        return _prepare_4d_attention_mask(attention_mask, input_tensor.dtype)
    elif len(attention_mask.size()) == 3:
        return _expand_mask_from_3d_mask(attention_mask, input_tensor.dtype)
    else:
        raise NotImplementedError(
            f"attention_mask of shape {attention_mask.size()} is not Implemented"
        )


def _expand_mask_from_3d_mask(mask: torch.Tensor, dtype: torch.dtype):
    """
    refer to: transformers/modeling_attn_mask_utils.py::AttentionMaskConverter._expand_mask
    Expands attention_mask from `[bsz, tgt_seq_len, src_seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    It's usually from block-wise attention for training with packed sequence
    """
    bsz, tgt_len, src_len = mask.size()
    expanded_mask = (
        mask[:, None, :, :].expand(bsz, 1, tgt_len, src_len).to(dtype).contiguous()
    )
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


# ===========================================================================
# B. Embedding & input preparation
# ===========================================================================
def _get_batch_size(input_ids, inputs_embeds):
    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    return batch_size


def _get_sequence_len(pad_token_id, in_, device):
    if pad_token_id is None:
        sequence_lengths = -1
    else:
        if in_ is not None:
            sequence_lengths = (torch.ne(in_, pad_token_id).sum(-1) - 1).to(device)
        else:
            sequence_lengths = -1
    return sequence_lengths


def _get_stacked_inputs_embeds(
    self,
    input_ids: torch.LongTensor = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
):
    # Converting tokens to look-up embeddings
    assert inputs_embeds is None
    inputs_embeds = self.model.embed_tokens(input_ids)
    if self.embed_dropout is not None:
        inputs_embeds = self.embed_dropout(inputs_embeds)

    if len(input_ids.shape) == 3:
        inputs_embeds = self.stacked_feat_agg(inputs_embeds)
        # [bz, seq, feat, dim] -> [bz, seq, dim]
        assert inputs_embeds.shape[:2] == input_ids.shape[:2]
        in_ = input_ids[:, :, 0]  # [bz, seq, num_feat] -> [bz, seq]
        # below `if` turns out to be useless
        if self.config.stack_method == "long":
            nonzero_feat = (input_ids != 0).sum(dim=-1, keepdim=True) + _EPSILON
            ratio = 1 / nonzero_feat.to(inputs_embeds.dtype)  # [bz, seq, 1]
            ratio = torch.clamp(ratio, max=1)
            inputs_embeds = inputs_embeds * ratio
    else:
        in_ = input_ids  # [bz, seq]
    input_ids = None
    return input_ids, inputs_embeds, in_


def _get_pos_type_embeds(embed_lookup: torch.nn.Module, pos_type: torch.Tensor):
    # Converting 3D position type to embeddings
    # sample_mask = _get_sample_lvl_mask(pos)  # [bz, 1]
    # pos_type = pos_type.clone().masked_fill_(sample_mask, 0)
    # ABOVE 2 lines pad `pos_type` with 0 => will slightly worsen fine-tune results, so remove it!
    pos_type = torch.clamp(pos_type, min=0)  # [bz, seq]
    type_embeds = embed_lookup(pos_type)  # [bz, seq, dim]
    return type_embeds


def transform_inputs_raw_embeds(self, inputs_raw_embeds, dtype):
    # Deal with input raw embeddings if any
    if self.config.embed_dim > 0:
        inputs_raw_embeds = inputs_raw_embeds.to(dtype)
        inputs_raw_embeds = self.embed_layernorm(inputs_raw_embeds)
        if self.raw_embed_dropout is not None:
            inputs_raw_embeds = self.raw_embed_dropout(inputs_raw_embeds)
        inputs_raw_embeds = self.embed_proj(inputs_raw_embeds)
        if len(inputs_raw_embeds.shape) == 4:  # [bsz, seq, seq, dim]
            inputs_raw_embeds = inputs_raw_embeds.sum(dim=-2)  # [bsz, seq, dim]
    else:
        inputs_raw_embeds = None
    return inputs_raw_embeds


# ===========================================================================
# C. Loss functions
# ===========================================================================
def _get_ce_loss(
    logits,
    labels,
    vocab_size,
    *,
    wgt=None,
    label_smoothing: float = 0,
    focal_gamma: float = 0,
):
    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)

    labels = labels.to(logits.device)
    if wgt is None:
        if focal_gamma > 0:
            loss_fct = utils_graphgpt.FocalLoss(focal_gamma)
        else:
            loss_fct = CrossEntropyLoss(label_smoothing=label_smoothing)
        loss = loss_fct(logits.float(), labels)
    else:
        if focal_gamma > 0:
            loss_fct = utils_graphgpt.FocalLoss(focal_gamma, reduction="none")
        else:
            loss_fct = CrossEntropyLoss(
                reduction="none", label_smoothing=label_smoothing
            )
        loss = loss_fct(logits.float(), labels)
        wgt1 = wgt.view(-1)
        loss = (loss * wgt1).float().sum() / (wgt1.float().sum() + _EPSILON)
    # convert logits to float before cross-entropy for molecule datasets like PCQM4M-v2, MOLPCBA and etc.
    # because when batch-size too large, ce with fp16 leads to no decrease of loss
    return loss


def _get_dlm_ce_loss(
    logits,
    labels,
    vocab_size,
    *,
    wgt,
):
    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)

    labels = labels.to(logits.device)
    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(logits.float(), labels)
    wgt1 = wgt.view(-1)
    loss = (loss * wgt1).float().sum()
    # convert logits to float before cross-entropy for molecule datasets like PCQM4M-v2, MOLPCBA and etc.
    # because when batch-size too large, ce with fp16 leads to no decrease of loss
    return loss


def _get_cl_logits_loss(
    cl_proj: torch.nn.Module,
    raw_hidden_states: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    inputs_embeds: torch.Tensor,
    in_: torch.Tensor,
    pad_token_id: int,
    world_size: int,
):
    device = raw_hidden_states.device
    batch_size = _get_batch_size(input_ids, inputs_embeds)
    assert pad_token_id is not None
    sequence_lengths = _get_sequence_len(pad_token_id, in_, device)
    bz_idx = torch.arange(batch_size, device=device)
    # [N, seq, dim] -> [N, dim]
    hidden_states = raw_hidden_states[bz_idx, sequence_lengths]
    logits = cl_proj(hidden_states)  # [bsz, dim]
    embeds = nn.functional.normalize(logits, dim=-1)

    left_idx = torch.arange(0, batch_size, 2, dtype=torch.long, device=device)
    left_embeds = embeds[left_idx].contiguous()

    right_idx = torch.arange(1, batch_size, 2, dtype=torch.long, device=device)
    right_embeds = embeds[right_idx].contiguous()

    loss = _dist_infonce(left_embeds, right_embeds, world_size=world_size)
    return loss, logits


# ===========================================================================
# D. Label/logit preparation
# ===========================================================================
def _prepare_for_logits_labels_per_seq_lvl(
    hidden_states: torch.FloatTensor,
    labels: torch.LongTensor,
    *,
    proj: torch.nn.Module,
):
    # i). obtain mask
    if labels is not None:
        mask = labels != -100  # [N, seq]
    else:  # for inference: pred all tokens
        mask = torch.ones(hidden_states.size()[:2], dtype=torch.bool)
    # ii). deal hidden states: mask and reshape
    # [N, seq, dim] -> [M, dim]
    hidden_states = hidden_states[mask]
    # [M, dim] -> [M, dim]
    hidden_states = proj(hidden_states)
    # iii). deal labels: mask and reshape
    if labels is not None:
        # [N, seq] -> [M]
        labels = labels[mask]
    # iv). obtain normalized wgt
    wgt = None
    if labels is not None:
        wgt = mask.float()  # [N, seq]
        wgt = wgt / (wgt.sum(dim=-1)[:, None] + _EPSILON)
        # [N, seq, n_token] -> [M]
        wgt = wgt[mask]
    return hidden_states, labels, wgt


def _prepare_for_stacked_feat_labels_per_mix_lvl(
    hidden_states: torch.FloatTensor,
    labels: torch.LongTensor,
    *,
    proj: torch.nn.Module,
):
    # per_seq_lvl vs per_feat_lvl::
    # seq == node-idx & node-feat & edge-feat in one column, i.e., tokens all masked in one column
    # feat == each token, i.e., either node-idx | node-feat | edge-feat
    # use seq_lvl ONLY when tokens in one column is masked together to SAVE compute power
    # Because in seq_lvl, `proj` is applied to each all-masked column (corresponding to non-masked labels)
    # after filtering out non-masked input columns
    # MIX-LVL: mix per-seq-lvl and per-feat-lvl
    dim = hidden_states.shape[-1]  # [N, seq, dim]
    # i). obtain mask
    if labels is not None:
        mask = labels != -100  # [N, seq, next_n]
        mask_m = mask.any(dim=-1)  # [N, seq]
        # mask_m -> seq-lvl mask
        # mask -> feat-lvl mask
        mask = mask[mask_m]  # [M, next_n]
    else:  # for inference: pred all tokens
        mask_m = torch.ones(hidden_states.size()[:2], dtype=torch.bool)
    # ii). deal hidden states: mask and reshape
    # [N, seq, dim] -> [M, dim]
    hidden_states = hidden_states[mask_m]
    # [M, dim] -> [M, dim*next_n]
    hidden_states = proj(hidden_states)
    # [M, dim*next_n] -> [M*next_n, dim]
    hidden_states = hidden_states.reshape((-1, dim))
    # iii). deal labels: mask and reshape
    if labels is not None:
        # 0.1 Converted by seq-lvl mask: [N, seq, next_n] -> [M, next_n]
        labels = labels[mask_m]
        # 0.2 Converted by feat-lvl mask: [M, next_n] -> [L]
        labels = labels[mask]
        # 0.3 Converted by feat-lvl mask: [M*next_n, dim] -> [L, dim]
        hidden_states = hidden_states[mask.reshape(-1)]
    return hidden_states, labels


def _prepare_for_per_feat_lvl(
    hidden_states: torch.FloatTensor,
    labels: torch.LongTensor,
    *,
    proj: torch.nn.Module,
    num_feat: int,
):
    # TODO: mix per-seq and per-feat lvl to boost speed
    batch_size, seq, dim = hidden_states.shape  # [N, seq, dim]
    hidden_states = proj(hidden_states)  # [N, seq, dim] -> [N, seq, dim*next_n]
    # i). obtain mask
    mask_m = labels != -100  # [N, seq, next_n]
    # ii). deal hidden states: reshape and then mask
    # [N, seq, dim*next_n] -> [N, seq, next_n, dim]
    hidden_states = hidden_states.reshape((batch_size, seq, num_feat, dim))
    # [N, seq, next_n, dim] -> [M, dim]
    hidden_states = hidden_states[mask_m]
    # iii). deal labels: mask
    # [N, seq, next_n] -> [M]
    labels = labels[mask_m]
    return hidden_states, labels, mask_m


def _prepare_for_stacked_feat_labels_per_feat_lvl(
    hidden_states: torch.FloatTensor,
    labels: torch.LongTensor,
    *,
    proj: torch.nn.Module,
    num_feat: int,
):
    hidden_states, labels, mask_m = _prepare_for_per_feat_lvl(
        hidden_states, labels, proj=proj, num_feat=num_feat
    )
    # iv). obtain normalized wgt
    wgt = mask_m.float()  # [N, seq, n_token]
    wgt = wgt / (wgt.sum(dim=-1).sum(dim=-1)[:, None, None] + _EPSILON)
    # [N, seq, n_token] -> [M]
    wgt = wgt[mask_m]
    return hidden_states, labels, wgt


def _prepare_for_stacked_feat_labels_wgt_per_feat_lvl(
    hidden_states: torch.FloatTensor,
    labels: torch.LongTensor,
    *,
    proj: torch.nn.Module,
    num_feat: int,
    wgt: torch.FloatTensor,
):
    hidden_states, labels, mask_m = _prepare_for_per_feat_lvl(
        hidden_states, labels, proj=proj, num_feat=num_feat
    )
    # iv). obtain wgt
    # [N, seq, n_token] -> [M]
    wgt = wgt[mask_m]
    return hidden_states, labels, wgt


def prepare_for_stacked_feat_labels(
    self,
    hidden_states: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    wgt: Optional[torch.FloatTensor] = None,
):
    if self.config.stack_method == "long":
        hidden_states, labels, wgt = _prepare_for_stacked_feat_labels_per_feat_lvl(
            hidden_states,
            labels,
            proj=self.n_token_proj,
            num_feat=self.config.next_n_token,
        )
    # the version below can save lots of GPU memory and boost speed
    if self.config.stack_method == "short" and wgt is None:
        hidden_states, labels = _prepare_for_stacked_feat_labels_per_mix_lvl(
            hidden_states,
            labels,
            proj=self.n_token_proj,
        )
    if self.config.stack_method == "short" and wgt is not None:
        seq = hidden_states.size(1)
        # [N] -> [N, seq, n_token]
        wgt = wgt[:, None, None].repeat(1, seq, self.config.next_n_token)
        hidden_states, labels, wgt = _prepare_for_stacked_feat_labels_wgt_per_feat_lvl(
            hidden_states,
            labels,
            proj=self.n_token_proj,
            num_feat=self.config.next_n_token,
            wgt=wgt,
        )
    return hidden_states, labels, wgt


# ===========================================================================
# E. 2D SMTP mask/noise
# ===========================================================================
def prepare_for_2d_smtp_inputs_labels(
    input_ids: torch.Tensor,
    node_idx: torch.Tensor,
    *,
    pos: torch.Tensor = None,
    smtp_2d_rate: float = 0,
    power: float = 0,  # 1 for linear mask scheduler
    replace_rate: float = 0,
    vocab: int = 2,
    global_2d_mask: bool = False,
    mask_token_id: int = 1,
    label_pad_token_id: int = -100,
):
    # input_ids: [bz, seq, feat], node_idx: [bz, seq], pos: [bz, seq, 3]
    bz, seq, feat = input_ids.size()
    device = input_ids.device
    eps = _EPSILON
    bz_idx = torch.arange(bz, device=device).view((-1, 1))
    raw_input_ids = input_ids.clone()

    # 1. get sample-lvl mask for samples for 2D-smtp
    sample_mask = (
        torch.rand((bz, 1, 1), dtype=torch.float32, device=device) < smtp_2d_rate
    )
    # re-calculate sample-mask, because some mols doesn't has pos, i.e., pos all 0's
    if pos is not None:
        sample_mask_pos = (pos.abs() < eps).all(dim=-1).all(dim=-1)
        sample_mask = sample_mask_pos[:, None, None] | sample_mask
    # 2. mask/pad samples for 2D-smtp
    mr_per_sample = torch.rand((bz, 1, 1), dtype=torch.float32, device=device)
    # apply polynomial mask on node-idx level
    mask_per_node = (
        torch.rand((bz, seq, feat), dtype=torch.float32, device=device)
        > mr_per_sample**power
    )
    # BELOW: ONLY mask samples of no pos info, i.e., pos all 0's
    if not global_2d_mask:
        mask_per_node = mask_per_node & sample_mask
    mask_per_token = mask_per_node[bz_idx, node_idx]  # look-up
    mask_per_token = mask_per_token & (input_ids > 0)
    labels = input_ids.clone().masked_fill_(~mask_per_token, label_pad_token_id)
    input_ids = input_ids.clone().masked_fill_(mask_per_token, mask_token_id)
    # 3. For [mask] tokens, randomly replace some with `token draw from vocab`
    # rnd_tokens = _get_uniform_rnd_tokens(raw_input_ids, vocab)
    rnd_tokens = _get_gaussian_rnd_tokens(raw_input_ids, vocab)
    replace_mask = (
        torch.rand((bz, seq, feat), dtype=torch.float32, device=device) < replace_rate
    )
    replace_mask = mask_per_token & replace_mask
    input_ids = input_ids * (~replace_mask).long() + rnd_tokens * replace_mask.long()
    return input_ids, labels


def _get_uniform_rnd_tokens(input_ids, vocab):
    bz, seq, feat = input_ids.size()
    device = input_ids.device
    rnd_tokens = torch.randint(1, vocab, size=(bz, seq, feat), device=device)
    rnd_tokens = rnd_tokens.masked_fill_(input_ids <= 0, 0)
    return rnd_tokens


def _get_gaussian_rnd_tokens(input_ids, vocab):
    bz, seq, feat = input_ids.size()
    device = input_ids.device
    std = 10  # math.sqrt((vocab**2-1)/12.0)
    token_shift = torch.randn(bz, seq, feat, dtype=torch.float32, device=device) * std
    token_shift = token_shift.round().long()
    token_shift = token_shift.masked_fill_(input_ids <= 0, 0)
    rnd_tokens = (input_ids + token_shift) % vocab
    return rnd_tokens


def _get_sample_lvl_mask(pos: torch.Tensor):
    mask = pos.abs() < _EPSILON  # [bz, seq, 3]
    # [bz, seq, 3] -> [bz, seq] -> [bz]
    sample_mask = mask.all(dim=-1).all(dim=-1, keepdim=True)  # [bz, 1]
    return sample_mask


# ===========================================================================
# F. 3D position token transforms
# ===========================================================================
def _mask_pos_in_node_lvl_on_schedule(
    noise, noisy_pos, noise_mask, pad_mask, node_idx, power
):
    mask_per_node, mask_per_coord, bz_idx = _preprocess_pos_smtp_masks(
        noisy_pos, power=power
    )  # power==0 <=> no 3d-SMTP
    mask_per_token = _get_mask_per_token_for_line(
        mask_per_node, mask_per_coord, pad_mask, bz_idx, node_idx, False
    )  # [bz, seq, 3]
    mask_per_token = mask_per_token[:, :, 0:1]
    noise = noise.masked_fill_(mask_per_token, 0)
    noisy_pos = noisy_pos.masked_fill_(mask_per_token, 0)
    noise_mask = noise_mask | mask_per_token
    return mask_per_token, noise, noisy_pos, noise_mask


def _mask_pad_pos_token_for_line(
    pos_tokens: torch.LongTensor,  # [bz, seq, 3/1]
    sample_mask: torch.Tensor,  # [bz, 1]
    pad_mask: torch.Tensor,  # [bz, seq]
    *,
    mask_per_token: torch.Tensor = None,  # [bz, seq, 3]
    mask_token_id: int = 1,
    pad_token_id: int = 0,
):
    # 1. fill the sample lvl: [mask]!
    # IF [pad], slightly worse results => for transfer learning where mols' pos it not available
    pos_tokens = pos_tokens.masked_fill_(sample_mask[:, :, None], mask_token_id)
    # 2. fill token-coordinate-lvl masked positions with [mask]
    if mask_per_token is not None:
        pos_tokens = pos_tokens.masked_fill_(mask_per_token, mask_token_id)
    # 3. fill the token lvl: [pad]
    pos_tokens = pos_tokens.masked_fill_(~pad_mask[:, :, None], pad_token_id)
    return pos_tokens


def _mask_raw_pos(
    noisy_pos: torch.Tensor,  # [bz, seq, 3]
    mask_per_token: torch.Tensor = None,  # [bz, seq, 3]
):
    if mask_per_token is not None:
        noisy_pos = noisy_pos.clone().masked_fill_(mask_per_token, 0)
    return noisy_pos


def transform_input_pos_via_line_token(self, pos, pos_type, mask_per_token=None):
    # pos -> [bz, seq, 3]  pos_type -> [bz, seq]  mask_per_token -> [bz, seq, 3/1]
    # 1. Deal with 3D positions
    # get sample-level mask
    sample_mask = _get_sample_lvl_mask(pos)
    pad_mask = pos_type > 0
    pos_tokens = _get_inputs_for_line_token(
        pos,
        num_bins=self.num_bins,
        range_min=self.range_min,
        range_max=self.range_max,
        sample_mask=sample_mask,
        pad_mask=pad_mask,
        mask_per_token=mask_per_token,
        pos_token_shift=self.pos_token_shift,
    )

    # 2 obtain pos-bins tokens's embedding
    pos_embeds = self.embed_pos_token(pos_tokens)  # [bz, seq, 3, dim]
    if self.embed_dropout is not None:
        pos_embeds = self.embed_dropout(pos_embeds)
    pos_embeds = self.pos_token_agg(pos_embeds)  # [bz, seq, dim]
    return pos_embeds


def _mask_pad_pos_token_for_cube(
    pos_tokens: torch.LongTensor,  # [bz, seq]
    sample_mask: torch.Tensor,  # [bz, 1]
    pad_mask: torch.Tensor,  # [bz, seq]
    mask_token_id: int = 1,
    pad_token_id: int = 0,
):
    # 1.31 fill the sample lvl: [mask]
    pos_tokens = pos_tokens.masked_fill_(sample_mask, mask_token_id)
    # 1.32 fill the token lvl: [pad]
    pos_tokens = pos_tokens.masked_fill_(~pad_mask, pad_token_id)
    return pos_tokens


def transform_input_pos_via_cube_token(self, pos, pos_type):
    # pos -> [bz, seq, 3]  pos_type -> [bz, seq]
    # 1. Deal with 3D positions
    # 1.1 obtain position-bins tokens after discretization
    # +2 to make sure min token is `2`: 0 for [pad], 1 for [mask]
    pos_tokens = discrete_pos(
        pos,
        self.num_bins,
        dict_bounds=self.dict_bounds,
        range_min=self.range_min,
        range_max=self.range_max,
    )
    # [bz, seq, 3] & [1, 3] -> [bz, seq, 3] -> [bz, seq]
    pos_tokens = (pos_tokens * self.idx_multiplier[None, :, :]).sum(dim=-1) + 2

    # 1.2 get sample-level mask
    sample_mask = _get_sample_lvl_mask(pos)

    # 1.3 fill position-bins tokens with `0`, the [pad] token, and `1`, the [mask] token
    pos_tokens = _mask_pad_pos_token_for_cube(
        pos_tokens, sample_mask, pad_mask=pos_type > 0
    )

    # 1.4 obtain pos-bins tokens's embedding
    pos_embeds = self.embed_pos_token(pos_tokens)  # [bz, seq, dim]
    if self.embed_dropout is not None:
        pos_embeds = self.embed_dropout(pos_embeds)
    return pos_embeds


def transform_input_pos_via_mix_token(self, pos, pos_type):
    # pos -> [bz, seq, 3]  pos_type -> [bz, seq]
    # 0. get sample-level mask
    sample_mask = _get_sample_lvl_mask(pos)
    pad_mask = pos_type > 0

    # 1. Deal with 3D positions
    # 1.1 obtain line-token
    pos_tokens = _get_inputs_for_line_token(
        pos,
        num_bins=self.num_bins_line,
        range_min=self.range_min,
        range_max=self.range_max,
        sample_mask=sample_mask,
        pad_mask=pad_mask,
        mask_per_token=None,
        pos_token_shift=self.pos_token_shift,
    )
    # turn tokens into embeds
    line_embeds = self.embed_line_token(pos_tokens)  # [bz, seq, 3, dim]
    if self.embed_dropout is not None:
        line_embeds = self.embed_dropout(line_embeds)
    line_embeds = self.line_token_agg(line_embeds)  # [bz, seq, dim]

    # 1.2 obtain cube-token
    # +2 to make sure min token is `2`: 0 for [pad], 1 for [mask]
    pos_tokens = discrete_pos(
        pos,
        self.num_bins_cube,
        dict_bounds=None,
        range_min=self.range_min,
        range_max=self.range_max,
    )  # [bz, seq, 3]
    # [bz, seq, 3] & [1, 3] -> [bz, seq, 3] -> [bz, seq]
    pos_tokens = (pos_tokens * self.idx_multiplier[None, :, :]).sum(dim=-1) + 2
    # mask and pad the cube-tokens
    pos_tokens = _mask_pad_pos_token_for_cube(pos_tokens, sample_mask, pad_mask)
    # turn tokens into embeds
    cube_embeds = self.embed_cube_token(pos_tokens)  # [bz, seq, dim]
    if self.embed_dropout is not None:
        cube_embeds = self.embed_dropout(cube_embeds)
    return line_embeds + cube_embeds


def prepare_pos_smtp_line_token_inputs_and_labels(
    self, pos, pos_type, node_idx, apply_denoise: bool = False
):
    # pos: [bz, seq, 3], pos_type: [bz, seq], node_idx: [bz, seq]
    noise_scale = self.smtp_3d_noise_scale  # set 0 to NOT DENOISE in SMTP pre-train
    # SMTP scheduler
    power = self.smtp_3d_power
    gt_rate = 0  # rate to unveil [mask] node's coord

    # 1. add noise, and get sample-level mask, pad-mask
    pos, noisy_pos, sample_mask, pad_mask, _, _ = _add_pos_noise_and_get_masks(
        pos, pos_type, noise_scale, node_idx
    )

    # 2. create SMTP mask
    mask_per_node, mask_per_coord, bz_idx = _preprocess_pos_smtp_masks(
        pos, power, gt_rate
    )
    mask_per_token = _get_mask_per_token_for_line(
        mask_per_node, mask_per_coord, pad_mask, bz_idx, node_idx, self.coord_lvl_mask
    )

    # 3. get pos input tokens and labels
    input_tokens = _get_inputs_for_line_token(
        noisy_pos,
        num_bins=self.num_bins,
        range_min=self.range_min,
        range_max=self.range_max,
        sample_mask=sample_mask,
        pad_mask=pad_mask,
        mask_per_token=mask_per_token,
        pos_token_shift=self.pos_token_shift,
    )
    labels = _get_labels_for_line_token(
        pos,
        num_bins=self.num_bins,
        range_min=self.range_min,
        range_max=self.range_max,
        sample_mask=sample_mask,
        pad_mask=pad_mask,
        mask_per_token=mask_per_token,
        pos_token_shift=self.pos_token_shift,
        denoise=apply_denoise,
    )

    # 4. obtain pos-bins tokens's embedding
    pos_embeds = self.embed_pos_token(input_tokens)  # [bz, seq, 3, dim]
    if self.embed_dropout is not None:
        pos_embeds = self.embed_dropout(pos_embeds)
    pos_embeds = self.pos_token_agg(pos_embeds)  # [bz, seq, dim]
    return pos_embeds, labels, _mask_raw_pos(noisy_pos, mask_per_token)


def _get_inputs_for_line_token(
    noisy_pos,
    *,
    num_bins,
    range_min,
    range_max,
    sample_mask,
    pad_mask,
    mask_per_token,
    pos_token_shift: torch.Tensor = None,
    mask_token_id: int = 1,
    pad_token_id: int = 0,
):
    # 1. get raw pos tokens
    input_tokens = discrete_pos(
        noisy_pos,
        num_bins,
        dict_bounds=None,
        range_min=range_min,
        range_max=range_max,
    )  # [bz, seq, 3]
    input_tokens = input_tokens + pos_token_shift + 2
    # 2. fill sample-lvl zeros' positions with [mask], and pad with [pad]
    input_tokens = _mask_pad_pos_token_for_line(
        input_tokens,
        sample_mask,
        pad_mask,
        mask_per_token=mask_per_token,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
    )
    return input_tokens


def _get_labels_for_line_token(
    clean_pos,
    *,
    num_bins,
    range_min,
    range_max,
    sample_mask,
    pad_mask,
    mask_per_token,
    pos_token_shift: torch.Tensor = None,
    denoise: bool = False,
    label_pad_token_id: int = -100,
):
    # deal with labels of pos tokens from clean pos
    labels = discrete_pos(
        clean_pos,
        num_bins,
        dict_bounds=None,
        range_min=range_min,
        range_max=range_max,
    )  # [bz, seq, 3]
    labels = labels + pos_token_shift
    # 1. fill sample-lvl zeros' positions with [label-pad]
    labels = labels.masked_fill_(sample_mask[:, :, None], label_pad_token_id)
    # 2. fill token-lvl non-masked positions with [label-pad]
    if denoise:
        labels = labels.masked_fill_(~pad_mask[:, :, None], label_pad_token_id)
    else:
        labels = labels.masked_fill_(~mask_per_token, label_pad_token_id)
    return labels


def prepare_pos_smtp_cube_token_inputs_and_labels(self, pos, pos_type, node_idx):
    # pos: [bz, seq, 3], pos_type: [bz, seq], node_idx: [bz, seq]
    noise_scale = self.smtp_3d_noise_scale  # set 0 to NOT DENOISE in SMTP pre-train
    # SMTP scheduler
    power = self.smtp_3d_power
    gt_rate = 0  # rate to unveil [mask] node's coord

    # 1. add noise, and get sample-level mask, pad-mask
    pos, noisy_pos, sample_mask, pad_mask, _, _ = _add_pos_noise_and_get_masks(
        pos, pos_type, noise_scale, node_idx
    )

    # 2. create SMTP mask
    mask_per_node, mask_per_coord, bz_idx = _preprocess_pos_smtp_masks(
        pos, power, gt_rate
    )
    mask_per_token = _get_mask_per_token_for_cube(
        mask_per_node, mask_per_coord, pad_mask, bz_idx, node_idx
    )

    # 3. get pos input tokens and labels
    input_tokens, labels = _get_inputs_and_labels_for_cube_token(
        noisy_pos,
        pos,
        num_bins=self.num_bins,
        range_min=self.range_min,
        range_max=self.range_max,
        sample_mask=sample_mask,
        pad_mask=pad_mask,
        mask_per_token=mask_per_token,
        idx_multiplier=self.idx_multiplier,
    )

    # 4. obtain pos-bins tokens's embedding
    pos_embeds = self.embed_pos_token(input_tokens)  # [bz, seq, dim]
    if self.embed_dropout is not None:
        pos_embeds = self.embed_dropout(pos_embeds)
    return pos_embeds, labels, _mask_raw_pos(noisy_pos, mask_per_token)


def _get_inputs_and_labels_for_cube_token(
    noisy_pos,
    clean_pos,
    *,
    num_bins,
    range_min,
    range_max,
    sample_mask,
    pad_mask,
    mask_per_token,
    idx_multiplier: torch.Tensor = None,
    mask_token_id: int = 1,
    pad_token_id: int = 0,
    label_pad_token_id: int = -100,
    **kwargs,
):
    # 3. get pos tokens, and then mask with 0
    input_tokens = discrete_pos(
        noisy_pos,
        num_bins,
        dict_bounds=None,
        range_min=range_min,
        range_max=range_max,
    )  # [bz, seq, 3]
    # 3.1 deal with inputs pos tokens
    # 3.11 fill sample-lvl zeros' positions with [mask]
    input_tokens = (input_tokens * idx_multiplier[None, :, :]).sum(dim=-1) + 2
    input_tokens = input_tokens.masked_fill_(sample_mask, mask_token_id)
    # 3.12 fill token-lvl masked positions with [mask]
    input_tokens = input_tokens.masked_fill_(mask_per_token, mask_token_id)
    # 3.13 fill token-lvl padded positions wtih [pad]
    input_tokens = input_tokens.masked_fill_(~pad_mask, pad_token_id)
    # 3.2 deal with labels of pos tokens from clean pos
    labels = discrete_pos(
        clean_pos,
        num_bins,
        dict_bounds=None,
        range_min=range_min,
        range_max=range_max,
    )  # [bz, seq, 3]
    # 3.21 fill sample-lvl zeros' positions with [label-pad]
    labels = (labels * idx_multiplier[None, :, :]).sum(dim=-1) + 2
    labels = labels.masked_fill_(sample_mask, label_pad_token_id)
    # 3.22 fill token-lvl non-masked positions with [label-pad]
    labels = labels.masked_fill_(~mask_per_token, label_pad_token_id)
    return input_tokens, labels


def prepare_pos_smtp_mix_token_inputs_and_labels(
    self, pos, pos_type, node_idx, apply_denoise: bool = False
):
    # pos: [bz, seq, 3], pos_type: [bz, seq], node_idx: [bz, seq]
    noise_scale = self.smtp_3d_noise_scale  # set 0 to NOT DENOISE in SMTP pre-train
    # SMTP scheduler
    power = self.smtp_3d_power
    gt_rate = 0  # rate to unveil [mask] node's coord

    # 1. add noise, and get sample-level mask, pad-mask
    pos, noisy_pos, sample_mask, pad_mask, _, _ = _add_pos_noise_and_get_masks(
        pos, pos_type, noise_scale, node_idx
    )

    # 2. create SMTP mask
    mask_per_node, mask_per_coord, bz_idx = _preprocess_pos_smtp_masks(
        pos, power, gt_rate
    )
    # a). For line token
    mask_per_token_line = _get_mask_per_token_for_line(
        mask_per_node, mask_per_coord, pad_mask, bz_idx, node_idx
    )
    # b). For cube token
    mask_per_token_cube = _get_mask_per_token_for_cube(
        mask_per_node, mask_per_coord, pad_mask, bz_idx, node_idx
    )

    # 3. get pos tokens, and then mask with 0
    # a). For line token
    input_tokens_line = _get_inputs_for_line_token(
        noisy_pos,
        num_bins=self.num_bins_line,
        range_min=self.range_min,
        range_max=self.range_max,
        sample_mask=sample_mask,
        pad_mask=pad_mask,
        mask_per_token=mask_per_token_line,
        pos_token_shift=self.pos_token_shift,
    )
    labels_line = _get_labels_for_line_token(
        pos,
        num_bins=self.num_bins_line,
        range_min=self.range_min,
        range_max=self.range_max,
        sample_mask=sample_mask,
        pad_mask=pad_mask,
        mask_per_token=mask_per_token_line,
        pos_token_shift=self.pos_token_shift,
        denoise=apply_denoise,
    )

    # b). For cube token
    input_tokens_cube, labels_cube = _get_inputs_and_labels_for_cube_token(
        noisy_pos,
        pos,
        num_bins=self.num_bins_cube,
        range_min=self.range_min,
        range_max=self.range_max,
        sample_mask=sample_mask,
        pad_mask=pad_mask,
        mask_per_token=mask_per_token_cube,
        idx_multiplier=self.idx_multiplier,
    )

    # 4. obtain pos-bins tokens's embedding
    line_embeds = self.embed_line_token(input_tokens_line)  # [bz, seq, 3, dim]
    cube_embeds = self.embed_cube_token(input_tokens_cube)  # [bz, seq, dim]
    if self.embed_dropout is not None:
        line_embeds = self.embed_dropout(line_embeds)
        cube_embeds = self.embed_dropout(cube_embeds)
    line_embeds = self.line_token_agg(line_embeds)  # [bz, seq, dim]
    pos_embeds = line_embeds + cube_embeds
    return (
        pos_embeds,
        (labels_line, labels_cube),
        _mask_raw_pos(noisy_pos, mask_per_token_line),
    )


def _preprocess_pos_smtp_masks(pos, power: float, gt_rate: float = 0):
    device = pos.device
    bz, seq, _ = pos.size()
    mr_per_sample = torch.rand((bz, 1, 1), dtype=torch.float32, device=device)

    if power == -2:  # arc-cosine scheduler
        mr_per_sample = torch.acos(mr_per_sample * 2 - 1) / torch.pi
    elif power == -1:  # cosine scheduler
        mr_per_sample = 0.5 * torch.cos(torch.pi * mr_per_sample) + 0.5
    else:  # polynomial scheduler
        mr_per_sample = mr_per_sample**power
    # apply polynomial mask on node-idx level
    mask_per_node = (
        torch.rand((bz, seq, 3), dtype=torch.float32, device=device) > mr_per_sample
    )
    mask_per_coord = (
        torch.rand((bz, seq, 3), dtype=torch.float32, device=device) > gt_rate
    )
    bz_idx = torch.arange(bz, device=device).view((-1, 1))
    return mask_per_node, mask_per_coord, bz_idx


def _add_pos_noise_and_get_masks(
    pos: torch.Tensor,
    pos_type: torch.Tensor,
    noise_scale: float,
    node_idx: torch.Tensor = None,
):
    # pos: [bz, seq, 3], noise_scale: scalar, node_idx: [bz, seq]
    # refer to: https://github.com/lsj2408/Transformer-M/blob/45647e143a5282e0e97117969396446084bcf1ab/Transformer-M/criterions/graph_prediction.py#L42
    bz = pos.size()[0]
    device = pos.device
    bz_idx = torch.arange(bz, device=device).view((-1, 1))
    # 1. get sample-level mask
    eps = _EPSILON
    mask = pos.abs() < eps  # [bz, seq, 3]
    pos = pos.masked_fill_(mask, 0)
    # 2. obtain noise-mask, new sample-mask directly from pos: will include mol with pos==0
    sample_mask = mask.all(dim=-1).all(dim=-1, keepdim=True)  # [bz, 1]
    if pos_type is not None:
        pad_mask = pos_type > 0  # [bz, seq]
        noise_mask = (~pad_mask) | sample_mask  # [bz, seq]
    else:
        noise_mask = mask.all(dim=-1)  # [bz, seq]
        pad_mask = ~noise_mask
    noise_mask = noise_mask[:, :, None]
    # 3. add noise to clean pos, and apply mask
    gnoise = torch.randn(pos.shape).to(pos) * noise_scale
    gnoise = gnoise[bz_idx, node_idx].contiguous()
    noise = gnoise.masked_fill_(noise_mask, 0.0)
    noisy_pos = pos + noise
    return pos, noisy_pos, sample_mask, pad_mask, noise, noise_mask


def _get_mask_per_token_for_line(
    mask_per_node: torch.Tensor,  # [bz, seq, 3]
    mask_per_coord: torch.Tensor,  # [bz, seq, 3]
    pad_mask: torch.Tensor,  # [bz, seq]
    bz_idx: torch.Tensor,  # [bz, 1]
    node_idx: torch.Tensor,  # [bz, seq]
    coord_lvl_mask: bool = False,
):
    # a). For line token
    if not coord_lvl_mask:
        mask_per_node = mask_per_node[:, :, 0:1]  # [bz, seq, 1]
    mask_per_node = mask_per_node & mask_per_coord
    # get mask on line token level through look-up
    mask_per_token = mask_per_node[bz_idx, node_idx]
    mask_per_token = mask_per_token & pad_mask[:, :, None]
    return mask_per_token


def _get_mask_per_token_for_cube(
    mask_per_node: torch.Tensor,  # [bz, seq, 3]
    mask_per_coord: torch.Tensor,  # [bz, seq, 3]
    pad_mask: torch.Tensor,  # [bz, seq]
    bz_idx: torch.Tensor,  # [bz, 1]
    node_idx: torch.Tensor,  # [bz, seq]
):
    # b). For cube token
    mask_per_node = mask_per_node[:, :, 0]  # [bz, seq]
    mask_per_node = mask_per_node & (mask_per_coord.any(dim=-1))
    # get mask on cube token level through look-up
    mask_per_token = mask_per_node[bz_idx, node_idx]
    mask_per_token = mask_per_token & pad_mask
    return mask_per_token
