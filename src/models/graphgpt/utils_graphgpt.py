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
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama import modeling_llama
from transformers.models.llama.configuration_llama import LlamaConfig
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


class LlamaMLP(modeling_llama.LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.mlp_pdrop)

    def forward(self, x):
        down_proj = super().forward(x)
        down_proj = self.dropout(down_proj)
        return down_proj


class LlamaAttention(modeling_llama.LlamaAttention):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        attn_output, attn_weights, past_key_value = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        # TODO: indicator, dropout added before outputing attn_output
        attn_output = self.resid_dropout(attn_output)
        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(modeling_llama.LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.self_attn
        del self.mlp
        if not getattr(config, "_flash_attn_2_enabled", False):
            self.self_attn = LlamaAttention(config=config)
        else:
            raise ValueError(
                "LlamaFlashAttention2 Not supported!!!"
            )  # LlamaFlashAttention2(config=config)
        self.mlp = LlamaMLP(config)


class LlamaModel(modeling_llama.LlamaModel):
    """
    Code modified from https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/modeling_llama.py
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        del self.layers
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds

        # TODO: indicator, dropout added before feeding to transformer layers
        hidden_states = self.drop(hidden_states)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _prepare_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        if self.config.causal_attention:
            return self._prepare_decoder_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )
        else:
            return self._prepare_encoder_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        if hasattr(super(), "_prepare_decoder_attention_mask"):
            # For transformer version 4.34.x
            return super()._prepare_decoder_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )
        elif hasattr(modeling_llama, "_prepare_4d_causal_attention_mask"):
            # For transformer version >= 4.35
            return modeling_llama._prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )
        elif hasattr(super(), "_update_causal_mask"):
            # For transformer version >= 4.38
            return super()._update_causal_mask(
                attention_mask, inputs_embeds
            )
        else:
            raise NotImplementedError("_prepare_decoder_attention_mask is NOT implemented!!!")

    def _prepare_encoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        return _prepare_4d_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )


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
