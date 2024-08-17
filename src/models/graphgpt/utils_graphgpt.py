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
from typing import List, Optional, Tuple, Union
import numpy as np

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama import modeling_llama
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.beit.modeling_beit import BeitDropPath
from transformers.utils.import_utils import is_torch_fx_available
from src.utils.attn_mask_utils import is_torch_greater_or_equal_than_1_13
from src.utils.attn_mask_utils import (
    _prepare_4d_causal_bi_attention_mask,
    _prepare_4d_attention_mask,
)

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
                init_values * torch.ones((config.hidden_size)), requires_grad=True
            )
            self.lambda_2 = nn.Parameter(
                init_values * torch.ones((config.hidden_size)), requires_grad=True
            )
        else:
            self.lambda_1, self.lambda_2 = None, None

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
            **kwargs,
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


class UniBiLlamaModel(modeling_llama.LlamaModel):
    """
    The only difference from `modeling_llama.LlamaModel` is the implementation of uni-bi-directional attn mask
    check https://aliyuque.antfin.com/james.zqf/ssqcu1/dexa1q0g8givelio?singleDoc# for details

    Code modified from https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/modeling_llama.py
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_bi: Optional[torch.Tensor] = None,
        boundary_mask_idx: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if not self.config.uni_bi_attn:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
            padding_mask = None
        else:
            if 0 in attention_mask:
                padding_mask = attention_mask
            else:
                padding_mask = None

        # TODO: indicator of customized uni-bi-directional attn mask
        attention_mask = _prepare_4d_causal_bi_attention_mask(
            attention_mask,
            attention_mask_bi,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            boundary_mask_idx=boundary_mask_idx,
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(
                            *inputs,
                            past_key_value,
                            output_attentions,
                            padding_mask=padding_mask,
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


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
