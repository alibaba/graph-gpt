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
Pre-training model classes: GraphGPTPretrainBase, GraphGPTPosPred.
"""
from copy import deepcopy
import torch
from torch import nn
import torch.distributed as dist
from typing import List, Optional, Tuple, Union
from transformers import LlamaForCausalLM
from transformers.models.llama import modeling_llama

from . import utils_graphgpt
from .modeling_common import (
    DoubleHeadsModelOutput,
    StackedFeatAggregation,
    POS_TYPE_mask_lookup,
    init_backbone,
    init_embed_dropout,
    init_stacked_feat_agg,
    resolve_forward_defaults,
)
from .modeling_helpers import (
    _update_causal_mask,
    _get_stacked_inputs_embeds,
    _get_pos_type_embeds,
    _get_ce_loss,
    _get_dlm_ce_loss,
    _get_cl_logits_loss,
    _prepare_for_stacked_feat_labels_per_feat_lvl,
    prepare_for_stacked_feat_labels,
    prepare_for_2d_smtp_inputs_labels,
    prepare_pos_smtp_line_token_inputs_and_labels,
    prepare_pos_smtp_cube_token_inputs_and_labels,
    prepare_pos_smtp_mix_token_inputs_and_labels,
)
from src.utils.mol_utils import DICT_range


class GraphGPTPretrainBase(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # 1. Transformer's backbone
        init_backbone(self, config)
        self.smtp_inside = config.smtp_inside  # apply mask in side model
        self.smtp_power = config.smtp_power
        # 1.1 Embedding dropout
        init_embed_dropout(self, config)
        # 1.2 Node/edge attributes stacking
        init_stacked_feat_agg(self, config, conditional=False)
        # 1.3 inputs got raw embed feature
        if config.embed_dim > 0:
            self.raw_embed_dropout = None
            if config.embed_pdrop > 0:
                self.raw_embed_dropout = nn.Dropout(p=config.embed_pdrop)
            self.embed_layernorm = modeling_llama.LlamaRMSNorm(
                config.embed_dim, eps=config.rms_norm_eps
            )
            std = self.config.initializer_range
            self.emb_mask_token = torch.nn.Parameter(
                torch.empty((1, 1, config.embed_dim)).normal_(mean=0.0, std=std),
                requires_grad=True,
            )
            self.embed_proj = nn.Linear(
                config.embed_dim, config.hidden_size, bias=False
            )
        # 2. Optimization objective
        print(
            f"Next/Masked-(1)-token-prediction changed to next/masked-{config.next_n_token}-tokens-prediction!"
        )
        if self.config.next_n_token > 1:
            self.n_token_proj = nn.Linear(
                config.hidden_size,
                config.hidden_size * config.next_n_token,
                bias=False,
            )
        else:
            self.n_token_proj = nn.Identity()
        # 2.1 generative or discriminative
        self.use_generative = self.config.use_generative
        self.use_discriminative = self.config.use_discriminative
        if self.use_generative and self.use_discriminative:
            self.ratio_dis = 0.5
        else:
            self.ratio_dis = 1
        if not self.use_generative:
            del self.lm_head
            del self.n_token_proj
        if self.use_discriminative:
            self.cl_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        try:
            self.world_size = dist.get_world_size()
        except:
            self.world_size = 1
        if self.use_discriminative:
            print(
                f"WORLD-SIZE for CL loss gathering and calculation: {self.world_size}"
            )
        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_raw_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        input_ids, inputs_embeds, in_ = _get_stacked_inputs_embeds(
            self, input_ids, inputs_embeds
        )

        # Deal with input raw embeddings if any
        if self.config.embed_dim > 0:
            inputs_raw_embeds = inputs_raw_embeds.to(inputs_embeds.dtype)
            # For inputs corresponding to -100 label, its embed shall multiply 1, i.e., the embed will be kept
            # [N, seq, next_n] -> [N, seq, 1]
            if self.smtp_inside:
                embed_mask = labels[:, :, 0:1] == -100
            else:
                embed_mask = (labels == -100).sum(dim=-1, keepdim=True).to(bool)
            # [N, seq, 1] * [1, 1, dim] -> [N, seq, dim]
            mask_part = (~embed_mask).to(inputs_embeds.dtype) * self.emb_mask_token
            # [N, seq, 1] * [N, seq, dim] -> [N, seq, dim]
            non_mask_part = embed_mask.to(inputs_embeds.dtype) * inputs_raw_embeds

            inputs_raw_embeds = non_mask_part + mask_part
            inputs_raw_embeds = self.embed_layernorm(inputs_raw_embeds)
            if self.raw_embed_dropout is not None:
                inputs_raw_embeds = self.raw_embed_dropout(inputs_raw_embeds)
            inputs_raw_embeds = self.embed_proj(inputs_raw_embeds)
            inputs_embeds = inputs_embeds + inputs_raw_embeds
        return input_ids, inputs_embeds, in_

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_raw_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        sample_wgt: Optional[torch.FloatTensor] = None,  # [bz], weight of sample
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[bool] = None,
    ) -> Union[Tuple, DoubleHeadsModelOutput]:
        output_attentions, output_hidden_states, return_dict, position_ids = (
            resolve_forward_defaults(
                self, output_attentions, output_hidden_states, return_dict, position_ids
            )
        )

        if self.smtp_inside:  # prepare the `input_ids` and `labels` for SMTP pre-train
            pos_deco = input_ids[:, :, self.config.stacked_feat :]  # [bz, seq, 4]
            node_idx = pos_deco[:, :, 2]  # [bz, seq]
            # [bz, seq, num_feats]
            input_ids = input_ids[:, :, : self.config.stacked_feat]

            input_ids, labels = prepare_for_2d_smtp_inputs_labels(
                input_ids,
                node_idx,
                smtp_2d_rate=1,
                power=self.smtp_power,
                replace_rate=0,
                vocab=self.config.vocab_size,
                global_2d_mask=False,
            )

        input_ids, inputs_embeds, in_ = self.prepare_inputs_embeds(
            input_ids, inputs_embeds, inputs_raw_embeds, labels
        )

        if not self.config.causal_attention:
            attention_mask = _update_causal_mask(self, attention_mask, inputs_embeds)
        outputs = self.model(
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

        raw_hidden_states = outputs[0]  # [N, seq, dim]

        gen_loss, gen_logits = None, None
        # 1. BELOW for generative pre-train, i.e., NTP/MTP/SMTP
        if self.use_generative:
            hidden_states, labels, wgt = prepare_for_stacked_feat_labels(
                self, raw_hidden_states, labels, sample_wgt
            )
            # hidden_states: wgt is None -> [bz*seq*next_n, dim]; wgt is not None -> [M, dim], M is total # of masked tokens
            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                if wgt is None:
                    loss = _get_ce_loss(
                        logits,
                        labels,
                        self.config.vocab_size,
                        focal_gamma=self.config.focal_gamma,
                    )
                else:
                    bz, seq, _ = raw_hidden_states.shape
                    loss = _get_dlm_ce_loss(
                        logits,
                        labels,
                        self.config.vocab_size,
                        wgt=wgt,
                    ) / (bz * seq * self.config.next_n_token)
            gen_loss, gen_logits = loss, logits
        # 2. BELOW for discriminative pre-train, i.e., CL
        dis_loss, dis_logits = None, None
        if self.use_discriminative:
            loss, logits = _get_cl_logits_loss(
                self.cl_proj,
                raw_hidden_states,
                input_ids,
                inputs_embeds,
                in_,
                self.config.pad_token_id,
                self.world_size,
            )
            dis_loss, dis_logits = loss * self.ratio_dis, logits

        if self.use_generative:
            head1_loss, head1_logits = gen_loss, gen_logits
            head2_loss, head2_logits = dis_loss, dis_logits
        else:
            head1_loss, head1_logits = dis_loss, dis_logits
            head2_loss, head2_logits = gen_loss, gen_logits
        return DoubleHeadsModelOutput(
            head1_loss=head1_loss,
            head1_logits=head1_logits,
            head2_loss=head2_loss,
            head2_logits=head2_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GraphGPTPosPred(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # 1. Transformer's backbone
        init_backbone(self, config)
        # 1.1 Embedding dropout
        init_embed_dropout(self, config)
        # 1.2 Node/edge attributes stacking
        init_stacked_feat_agg(self, config, conditional=True)
        # 1.3 embed 3D position type, 0~4: 0->pad, 1->(0,0,0), 2->(0,0,z), 3->(0,y,z), 4->(x,y,z)
        self.embed_pos_type = nn.Embedding(5, config.hidden_size, padding_idx=0)

        # 2. set-up 2D-SMTP pre-train
        # randomly pick some sample, then randomly mask their attrs for 2d-smtp
        self.smtp_2d_rate = config.pt_smtp_2d_rate
        # `smtp_2d_replace_rate` -> rate of replacing [mask] with randomly drawn tokens => inject noise to 2D tokens
        self.smtp_2d_replace_rate = config.smtp_2d_replace_rate
        # `sep_2d3d_inputs` -> 3D/2D-SMTP use separate samples: 2D-SMTP samples' pos set to 0
        self.sep_2d3d_inputs = config.sep_2d3d_inputs
        # `global_2d_mask` -> apply 2d mask on all samples, including 2D/3D
        self.global_2d_mask = config.global_2d_mask
        if self.config.next_n_token > 1:
            self.n_token_proj = nn.Linear(
                config.hidden_size,
                config.hidden_size * config.next_n_token,
                bias=False,
            )
        else:
            self.n_token_proj = nn.Identity()

        # 3. prepare for 3D pre-train
        # 3.1 init general params
        self.problem_type = config.pt_problem_type
        self.smtp_3d_power = config.smtp_3d_power
        self.smtp_3d_noise_scale = config.smtp_3d_noise_scale
        # coord_lvl_mask: mask position in per coordinate level instead of position level
        # (i.e., 3 coordinates together for one position)
        self.coord_lvl_mask = config.coord_lvl_mask
        self.num_bins = self.config.pos_bins = config.pt_num_bins
        # `apply_denoise == True` means un-masked noisy coord will be predicted using clean coord as the target!
        self.apply_denoise = config.apply_denoise
        # 3.2 init for different pre-train objectives
        self.label_smoothing = config.label_smoothing
        self.config.pos_agg_method = self.pos_agg_method = config.pt_pos_agg_method
        if self.problem_type == "pos-smtp-line":
            self._init_line_token_transform()
        elif self.problem_type == "pos-smtp-cube":
            self._init_cube_token_transform()
        elif self.problem_type == "pos-smtp-mix":
            self._init_mix_token_transform()

        # 4. raw-pos input projection
        self.use_pos_proj = config.use_pos_proj
        if self.use_pos_proj:
            self.in_pos_layernorm = modeling_llama.LlamaRMSNorm(
                3, eps=config.rms_norm_eps
            )
            self.in_pos_proj = nn.Linear(3, config.hidden_size, bias=False)

        # 5. SMTP loss aggregation
        self.loss_agg = config.loss_agg
        self.config.pos_range = pos_range = config.pt_pos_range

        print(f"problem_type: {self.problem_type}, num_bins: {self.num_bins}")
        self.register_buffer(
            "pos_type_mask_lookup", POS_TYPE_mask_lookup, persistent=False
        )  # pos_type_mask_lookup
        range_min, range_max = DICT_range[pos_range]
        print(f"pos_range: {pos_range} -> {(range_min, range_max)}")
        self.register_buffer("range_min", range_min, persistent=False)
        self.register_buffer("range_max", range_max, persistent=False)

        # 6. whether to add CL-loss
        self.use_discriminative = config.pt_use_discriminative
        if self.use_discriminative:
            self.ratio_dis = 1
            self.cl_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        try:
            self.world_size = dist.get_world_size()
        except:
            self.world_size = 1
        print(f"WORLD-SIZE for CL loss gathering and calculation: {self.world_size}")
        # Initialize weights and apply final processing
        self.post_init()

    def _init_line_token_transform(self):
        # discretize the (noisy) 3D position inputs
        # 1. set-up input
        if self.pos_agg_method == "sum":
            pos_token_shift = torch.tensor(
                [[[0, self.num_bins, self.num_bins * 2]]], dtype=torch.int64
            )
            pos_vocab = self.num_bins * 3 + 2
        else:
            pos_token_shift = torch.tensor([[[0, 0, 0]]], dtype=torch.int64)
            pos_vocab = self.num_bins + 2
        self.register_buffer("pos_token_shift", pos_token_shift, persistent=False)
        self.embed_pos_token = nn.Embedding(
            pos_vocab, self.config.hidden_size, padding_idx=0
        )  # [pad]->0, [mask]->1
        tmp_config = deepcopy(self.config)
        tmp_config.stacked_feat = 3
        tmp_config.stacked_feat_agg_method = self.pos_agg_method  # sum|gated
        self.pos_token_agg = StackedFeatAggregation(tmp_config)

        # 2. set-up task proj & head
        self.pos_vocab = pos_vocab - 2
        self.pos_bins_proj = nn.Linear(
            self.config.hidden_size, 3 * self.config.hidden_size, bias=False
        )
        self.pos_bins_head = nn.Linear(
            self.config.hidden_size, self.pos_vocab, bias=False
        )
        if hasattr(self, "raw_embed_dropout"):
            del self.raw_embed_dropout
        if hasattr(self, "embed_layernorm"):
            del self.embed_layernorm
        if hasattr(self, "embed_proj"):
            del self.embed_proj

    def _init_cube_token_transform(self):
        self.pos_vocab = self.num_bins**3 + 2
        self.pos_bins_proj = nn.Identity()
        self.pos_bins_head = nn.Linear(
            self.config.hidden_size, self.pos_vocab, bias=False
        )
        self.register_buffer(
            "idx_multiplier",
            torch.tensor(
                [[self.num_bins**2, self.num_bins**1, self.num_bins**0]],
                dtype=torch.int64,
            ),
            persistent=False,
        )
        if self.problem_type == "pos-smtp-cube":
            self.embed_pos_token = nn.Embedding(
                self.pos_vocab, self.config.hidden_size, padding_idx=0
            )  # [pad]->0, [mask]->1
            # tie weights if needed: in `modeling_utils.py`
            print(f"Tie weights: {self.pos_bins_head} and {self.embed_pos_token}")
            self._tie_or_clone_weights(self.pos_bins_head, self.embed_pos_token)

    def _init_mix_token_transform(self):
        # a). deal with line tokens, i.e., each pos with 3 coordinates form 3 tokens
        self.num_bins_line = self.config.pt_num_bins_line
        if self.pos_agg_method == "sum":
            pos_token_shift = torch.tensor(
                [[[0, self.num_bins_line, self.num_bins_line * 2]]],
                dtype=torch.int64,
            )
            self.line_token_vocab = self.num_bins_line * 3 + 2
        else:
            pos_token_shift = torch.tensor([[[0, 0, 0]]], dtype=torch.int64)
            self.line_token_vocab = self.num_bins_line + 2
        self.register_buffer("pos_token_shift", pos_token_shift, persistent=False)
        tmp_config = deepcopy(self.config)
        tmp_config.stacked_feat = 3
        tmp_config.stacked_feat_agg_method = self.pos_agg_method  # sum|gated

        self.embed_line_token = nn.Embedding(
            self.line_token_vocab, self.config.hidden_size, padding_idx=0
        )  # [pad]->0, [mask]->1
        self.line_token_agg = StackedFeatAggregation(tmp_config)

        self.line_token_proj = nn.Linear(
            self.config.hidden_size, 3 * self.config.hidden_size, bias=False
        )
        self.line_token_head = nn.Linear(
            self.config.hidden_size, self.line_token_vocab, bias=False
        )

        # b). deal with cube tokens, i.e., each pos with 3 coordinates form one token
        self.num_bins_cube = self.config.pt_num_bins_cube
        self.cube_token_vocab = self.num_bins_cube**3 + 2

        self.embed_cube_token = nn.Embedding(
            self.cube_token_vocab, self.config.hidden_size, padding_idx=0
        )  # [pad]->0, [mask]->1
        self.register_buffer(
            "idx_multiplier",
            torch.tensor(
                [
                    [
                        self.num_bins_cube**2,
                        self.num_bins_cube**1,
                        self.num_bins_cube**0,
                    ]
                ],
                dtype=torch.int64,
            ),
            persistent=False,
        )

        self.cube_token_proj = nn.Linear(
            self.config.hidden_size, self.config.hidden_size, bias=False
        )
        self.cube_token_head = nn.Linear(
            self.config.hidden_size, self.cube_token_vocab, bias=False
        )

        # tie weights if needed: in `modeling_utils.py`
        print(f"Tie weights: {self.cube_token_head} and {self.embed_cube_token}")
        self._tie_or_clone_weights(self.cube_token_head, self.embed_cube_token)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_raw_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[bool] = None,
    ) -> Union[Tuple, DoubleHeadsModelOutput]:
        output_attentions, output_hidden_states, return_dict, position_ids = (
            resolve_forward_defaults(
                self, output_attentions, output_hidden_states, return_dict, position_ids
            )
        )

        # 0. prepare the inputs related to positions
        pos_deco = input_ids[:, :, self.config.stacked_feat :]  # [bz, seq, 4]
        pos_type = pos_deco[:, :, 0]  # [bz, seq]
        node_idx = pos_deco[:, :, 2]  # [bz, seq]
        # [bz, seq, num_feats]
        input_ids = input_ids[:, :, : self.config.stacked_feat]
        raw_pos = inputs_raw_embeds[:, :, :3].contiguous()  # [bz, seq, 3]

        if self.training:
            smtp_2d_rate = self.smtp_2d_rate
            replace_rate = self.smtp_2d_replace_rate
            apply_denoise = self.apply_denoise
            global_2d_mask = self.global_2d_mask
        else:
            # during eval, only mols with pos all 0's being eval
            smtp_2d_rate = 0
            replace_rate = 0
            apply_denoise = False
            global_2d_mask = False
        if self.use_discriminative:
            raw_pos = utils_graphgpt.apply_sample_lvl_mask_alternative(raw_pos)
            smtp_2d_rate = 0
        elif self.sep_2d3d_inputs:
            raw_pos = utils_graphgpt.apply_sample_lvl_mask_pos(raw_pos, smtp_2d_rate)
            # 2D-SMTP samples is picked together with pos mask, so no need to pick again in the following process
            smtp_2d_rate = 0
            # IF 2D-SMTP samples' input contains noisy 3D-tokens, 2D-SMTP loss won't converge
            # So it's better to set their pos to be 0's
        power_2d = 1
        input_ids, labels_2d = prepare_for_2d_smtp_inputs_labels(
            input_ids,
            node_idx,
            pos=raw_pos,
            smtp_2d_rate=smtp_2d_rate,
            power=power_2d,
            replace_rate=replace_rate,
            vocab=self.config.vocab_size,
            global_2d_mask=global_2d_mask,
        )

        input_ids, inputs_embeds, in_ = _get_stacked_inputs_embeds(
            self, input_ids, inputs_embeds
        )
        inputs_embeds += _get_pos_type_embeds(self.embed_pos_type, pos_type)
        if self.problem_type == "pos-smtp-line":
            (
                pos_embeds,
                labels,
                masked_noisy_pos,
            ) = prepare_pos_smtp_line_token_inputs_and_labels(
                self, raw_pos, pos_type, node_idx, apply_denoise
            )  # [bz, seq, dim]     [bz, seq, 3]
        elif self.problem_type == "pos-smtp-cube":
            (
                pos_embeds,
                labels,
                masked_noisy_pos,
            ) = prepare_pos_smtp_cube_token_inputs_and_labels(
                self, raw_pos, pos_type, node_idx
            )  # [bz, seq, dim]     [bz, seq, 3]
        elif self.problem_type == "pos-smtp-mix":
            (
                pos_embeds,
                labels,
                masked_noisy_pos,
            ) = prepare_pos_smtp_mix_token_inputs_and_labels(
                self, raw_pos, pos_type, node_idx, apply_denoise
            )  # [bz, seq, dim]     [bz, seq, 3]
        inputs_embeds += pos_embeds
        if self.use_pos_proj:
            noisy_pos_embeds = self.in_pos_layernorm(
                masked_noisy_pos.to(inputs_embeds.dtype)
            )
            noisy_pos_embeds = self.in_pos_proj(noisy_pos_embeds)
            if self.embed_dropout is not None:
                noisy_pos_embeds = self.embed_dropout(noisy_pos_embeds)
            inputs_embeds += noisy_pos_embeds

        # 1. run backbone transformer
        if not self.config.causal_attention:
            attention_mask = _update_causal_mask(self, attention_mask, inputs_embeds)
        outputs = self.model(
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

        raw_hidden_states = outputs[0]  # [N, seq, dim]

        # 2. DEAL with pre-train tasks and losses
        # 2.1 calculate 2D SMTP loss if needed
        aux_loss = None
        (
            hidden_states,
            labels_2d,
            wgt_2d,
        ) = _prepare_for_stacked_feat_labels_per_feat_lvl(
            raw_hidden_states,
            labels_2d,
            proj=self.n_token_proj,
            num_feat=self.config.next_n_token,
        )
        logits = self.lm_head(hidden_states)
        if labels_2d is not None:
            vocab_size = self.config.vocab_size
            wgt_2d = None if self.loss_agg == "token-lvl" else wgt_2d
            aux_loss = _get_ce_loss(
                logits,
                labels_2d,
                vocab_size,
                wgt=wgt_2d,
                focal_gamma=self.config.focal_gamma,
            )

        # 2.2 calculate 3D pos related loss
        loss = None
        if labels is not None:
            if self.problem_type in ("pos-smtp-line", "pos-smtp-cube"):
                dim = raw_hidden_states.shape[-1]
                # hidden_states = self.pos_bins_proj(raw_hidden_states)  # [bz, seq, dim*3]
                (
                    hidden_states,
                    labels,
                    wgt_3d,
                ) = _prepare_for_stacked_feat_labels_per_feat_lvl(
                    raw_hidden_states,
                    labels,
                    proj=self.pos_bins_proj,
                    num_feat=3 if self.problem_type == "pos-smtp-line" else 1,
                )
                hidden_states = hidden_states.view((-1, dim))
                logits = self.pos_bins_head(hidden_states)
                vocab_size = self.pos_vocab
                wgt_3d = None if self.loss_agg == "token-lvl" else wgt_3d
                loss = _get_ce_loss(
                    logits,
                    labels,
                    vocab_size,
                    wgt=wgt_3d,
                    focal_gamma=self.config.focal_gamma,
                )
            elif self.problem_type == "pos-smtp-mix":
                dim = raw_hidden_states.shape[-1]
                # a). line-token
                hidden_states = self.line_token_proj(
                    raw_hidden_states
                )  # [bz, seq, dim*3]
                hidden_states = hidden_states.view((-1, dim))
                line_logits = self.line_token_head(hidden_states)
                vocab_size = self.line_token_vocab
                line_loss = _get_ce_loss(line_logits, labels[0], vocab_size)
                # b). cube-token
                hidden_states = self.cube_token_proj(
                    raw_hidden_states
                )  # [bz, seq, dim]
                hidden_states = hidden_states.view((-1, dim))
                cube_logits = self.cube_token_head(hidden_states)
                vocab_size = self.cube_token_vocab
                cube_loss = _get_ce_loss(cube_logits, labels[1], vocab_size)

                # mix experiments
                loss = line_loss  # line_loss
                aux_loss = cube_loss  # cube_loss
                logits = line_logits  # line_logits  cube_logits

        # 2.3 CL loss
        dis_loss, dis_logits = None, None
        if self.use_discriminative:
            dis_loss, dis_logits = _get_cl_logits_loss(
                self.cl_proj,
                raw_hidden_states,
                input_ids,
                inputs_embeds,
                in_,
                self.config.pad_token_id,
                self.world_size,
            )
            if aux_loss is None:
                aux_loss = dis_loss * self.ratio_dis
            else:
                aux_loss += dis_loss * self.ratio_dis

        return DoubleHeadsModelOutput(
            head1_loss=loss,
            head1_logits=logits,
            head2_loss=aux_loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
