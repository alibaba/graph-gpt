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
import math
import numpy as np
from copy import deepcopy
from functools import partial
import torch
from dataclasses import dataclass
from torch import nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, L1Loss
from typing import List, Optional, Tuple, Union
from transformers.utils import ModelOutput
from transformers import LlamaPreTrainedModel, LlamaForCausalLM
from transformers.models.llama import modeling_llama
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from . import utils_graphgpt
from src.utils.modules_utils import MLP
from src.utils.loss_utils import auc_loss, _dist_infonce
from src.utils.mol_utils import discrete_pos, DICT_range
from src.utils.attn_mask_utils import _prepare_4d_bi_causal_attention_mask
from src.utils.tokenizer_utils import MOL_ENERGY_BIN_LEN, MOL_ENERGY_SCALE


_prepare_4d_bi_causal_attention_mask = partial(
    _prepare_4d_bi_causal_attention_mask, MOL_ENERGY_BIN_LEN - 1
)  # -1 because the last binary digit won't be used in input_ids, but ONLY in labels
_EPSILON = 1e-7
# POS_TYPE_mask_lookup is only for molecular 3D data
POS_TYPE_mask_lookup = torch.tensor(
    [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.int64
).to(bool)


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
    attentions: Optional[Tuple[torch.FloatTensor]] = None


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


class GraphGPTCausal(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # 1. Transformer's backbone
        LlamaModel = (
            utils_graphgpt.LlamaModel
            if _use_dropout(self.config)
            else modeling_llama.LlamaModel
        )
        if not config.causal_attention:
            print(
                f"\nMonkey Patch {LlamaModel.__name__}'s method `_update_causal_mask`!\n"
            )
            LlamaModel._update_causal_mask = _update_causal_mask
        self.model = LlamaModel(config)
        self.smtp_inside = config.smtp_inside  # apply mask in side model
        self.smtp_power = 1
        # 1.1 Embedding dropout
        self.embed_dropout = None
        if config.embed_pdrop > 0:
            self.embed_dropout = nn.Dropout(p=config.embed_pdrop)
        # 1.2 Node/edge attributes stacking
        if config.stack_method in {"short", "long"}:
            self.stacked_feat_agg = StackedFeatAggregation(config)
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
        if config.stack_method in {"short", "long"}:
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
        self.use_generative = True
        self.use_discriminative = False
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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[bool] = None,
    ) -> Union[Tuple, DoubleHeadsModelOutput]:
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

        if self.smtp_inside:  # prepare the `input_ids` and `labels` for SMTP pre-train
            pos_deco = input_ids[:, :, self.config.stacked_feat :]  # [bz, seq, 4]
            node_idx = pos_deco[:, :, 2]  # [bz, seq]
            # [bz, seq, num_feats]
            input_ids = input_ids[:, :, : self.config.stacked_feat]

            # print(f"[DEBUG] before -> input_ids:\n{input_ids}")
            input_ids, labels = prepare_for_2d_smtp_inputs_labels(
                input_ids,
                node_idx,
                smtp_2d_rate=1,
                power=self.smtp_power,
                replace_rate=0,
                vocab=self.config.vocab_size,
                global_2d_mask=False,
            )
            # print(f"[DEBUG] after -> input_ids:\n{input_ids}\nlabels:\n{labels}")

        input_ids, inputs_embeds, in_ = self.prepare_inputs_embeds(
            input_ids, inputs_embeds, inputs_raw_embeds, labels
        )

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
            hidden_states, labels = prepare_for_stacked_feat_labels(
                self, raw_hidden_states, labels
            )
            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                loss = _get_ce_loss(
                    logits,
                    labels,
                    self.config.vocab_size,
                    focal_gamma=self.config.focal_gamma,
                )
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
        LlamaModel = (
            utils_graphgpt.LlamaModel
            if _use_dropout(self.config)
            else modeling_llama.LlamaModel
        )
        if not config.causal_attention:
            print(
                f"\nMonkey Patch {LlamaModel.__name__}'s method `_update_causal_mask`!\n"
            )
            LlamaModel._update_causal_mask = _update_causal_mask
        self.model = LlamaModel(config)
        # 1.1 Embedding dropout
        if config.embed_pdrop > 0:
            self.embed_dropout = nn.Dropout(p=config.embed_pdrop)
        else:
            self.embed_dropout = None
        # 1.2 Node/edge attributes stacking
        if config.stack_method in {"short", "long"}:
            self.stacked_feat_agg = StackedFeatAggregation(config)
        # 1.3 embed 3D position type, 0~4: 0->pad, 1->(0,0,0), 2->(0,0,z), 3->(0,y,z), 4->(x,y,z)
        self.embed_pos_type = nn.Embedding(5, config.hidden_size, padding_idx=0)

        # 2. set-up 2D-SMTP pre-train
        # randomly pick some sample, then randomly mask their attrs for 2d-smtp
        self.smtp_2d_rate = 0.1
        # `smtp_2d_replace_rate` -> rate of replacing [mask] with randomly drawn tokens => inject noise to 2D tokens
        # turns out to making fine-tune results worse
        self.smtp_2d_replace_rate = 0
        # `sep_2d3d_inputs` -> 3D/2D-SMTP use separate samples: 2D-SMTP samples' pos set to 0
        self.sep_2d3d_inputs = True
        # `global_2d_mask` -> apply 2d mask on all samples, including 2D/3D
        self.global_2d_mask = False
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
        # pos-smtp-line|pos-smtp-cube|pos-smtp-mix  -> pos-smtp-mix means 2D-smtp & 3D-smtp together
        # turns out denoise-pos-smtp-line to be the best!!!
        self.problem_type = "pos-smtp-line"
        self.smtp_3d_power = 1  # polynomial: 0.75/1/1.25, cosine: -1, arccosine: -2
        self.smtp_3d_noise_scale = 0.2
        # coord_lvl_mask: mask position in per coordinate level instead of position level
        # (i.e., 3 coordinates together for one position)
        self.coord_lvl_mask = True
        self.num_bins = self.config.pos_bins = 1024
        # `apply_denoise == True` means un-masked noisy coord will be predicted using clean coord as the target!
        self.apply_denoise = False
        # 3.2 init for different pre-train objectives
        self.label_smoothing = 0
        self.config.pos_agg_method = self.pos_agg_method = "gated"
        if self.problem_type == "pos-smtp-line":
            self._init_line_token_transform()
        elif self.problem_type == "pos-smtp-cube":
            self._init_cube_token_transform()
        elif self.problem_type == "pos-smtp-mix":
            self._init_mix_token_transform()

        # 4. raw-pos input projection
        self.use_pos_proj = False
        if self.use_pos_proj:
            self.in_pos_layernorm = modeling_llama.LlamaRMSNorm(
                3, eps=config.rms_norm_eps
            )
            self.in_pos_proj = nn.Linear(3, config.hidden_size, bias=False)

        # 5. SMTP loss aggregation
        self.loss_agg = "token-lvl"  # token-lvl|sample-lvl
        self.config.pos_range = pos_range = "p1p"

        print(f"problem_type: {self.problem_type}, num_bins: {self.num_bins}")
        self.register_buffer(
            "pos_type_mask_lookup", POS_TYPE_mask_lookup, persistent=False
        )  # pos_type_mask_lookup
        range_min, range_max = DICT_range[pos_range]
        print(f"pos_range: {pos_range} -> {(range_min, range_max)}")
        self.register_buffer("range_min", range_min, persistent=False)
        self.register_buffer("range_max", range_max, persistent=False)

        # 6. whether to add CL-loss
        self.use_discriminative = False
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
        # tie weights if needed: in `modeling_utils.py`
        # self._tie_or_clone_weights(self.pos_bins_head, self.embed_pos_token)
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
        self.num_bins_line = 256
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
        self.num_bins_cube = 32
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
            # print(f"[DEBUG] before mask -> raw_pos: {raw_pos}")
            raw_pos = utils_graphgpt.apply_sample_lvl_mask_alternative(raw_pos)
            smtp_2d_rate = 0
            # print(f"[DEBUG] after mask -> raw_pos: {raw_pos}")
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
        # print(f"[DEBUG] input_ids:\n{input_ids}\nlabels_2d:\n{labels_2d}")

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

        # print(f"[DEBUG] labels_3d:\n{labels}")
        # print(
        #     f"[Debug] pos_bins_head:\n{self.pos_bins_head.weight.data}\n{self.pos_bins_head.weight.data.shape}\n"
        #     f"[Debug] embed_pos_token:\n{self.embed_pos_token.weight.data}\n{self.embed_pos_token.weight.data.shape}\n\n\n"
        # )
        # 1. run backbone transformer
        # position_ids = None
        # print(f"[DEBUG] position_ids:\n{position_ids}")
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


class GraphGPTTaskModel(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # 1. Transformer's backbone
        LlamaModel = (
            utils_graphgpt.LlamaModel
            if _use_dropout(self.config)
            else modeling_llama.LlamaModel
        )
        if not config.causal_attention:
            print(
                f"\nMonkey Patch {LlamaModel.__name__}'s method `_update_causal_mask`!\n"
            )
            LlamaModel._update_causal_mask = _update_causal_mask
        self.model = LlamaModel(config)
        # 1.1 Embedding dropout
        if config.embed_pdrop > 0:
            self.embed_dropout = nn.Dropout(p=config.embed_pdrop)
        else:
            self.embed_dropout = None
        # 1.2 Node/edge attributes stacking
        if config.stack_method in {"short", "long"}:
            self.stacked_feat_agg = StackedFeatAggregation(config)
        # 1.3 inputs got raw embed feature
        if config.embed_dim > 0:
            self.raw_embed_dropout = None
            if config.embed_pdrop > 0:
                self.raw_embed_dropout = nn.Dropout(p=config.embed_pdrop)
            self.embed_layernorm = modeling_llama.LlamaRMSNorm(
                config.embed_dim, eps=config.rms_norm_eps
            )
            self.embed_proj = nn.Linear(
                config.embed_dim, config.hidden_size, bias=False
            )  # set bias=True to serve as the default when `embed==0`
        # 2. Init for SequenceClassification, refer to `LlamaForSequenceClassification`
        bias = self.config.problem_type == "regression"
        self.num_labels = config.num_labels
        if len(self.config.mlp) > 0:
            self.score = MLP(
                config.hidden_size,
                self.num_labels,
                mlp=self.config.mlp,
                hidden_act=self.config.hidden_act,
                dropout=self.config.dropout,
                bias=bias,
            )
        else:
            self.score = nn.Linear(config.hidden_size, self.num_labels, bias=bias)
        self.pos_weight = None

        self.pooling_method = config.pooling_method  # "last|sum|mean"
        print(f"Pooling in last layer is {self.pooling_method}!")
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_raw_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        input_ids, inputs_embeds, in_ = _get_stacked_inputs_embeds(
            self, input_ids, inputs_embeds
        )

        inputs_raw_embeds = transform_inputs_raw_embeds(
            self, inputs_raw_embeds, inputs_embeds.dtype
        )
        if inputs_raw_embeds is not None:
            inputs_embeds = inputs_embeds + inputs_raw_embeds
        return input_ids, inputs_embeds, in_

    def get_logits_for_token_lvl_task(
        self, batch_size, hidden_states, logits, pooled_logits, cls_idx
    ):
        if self.config.loss_type == "token_ce_intra":
            inv_temperature = 20
            hidden_states = nn.functional.normalize(hidden_states, dim=-1)
            # get the slice index of intra-instance classes
            # [bz] -> [bz, 1] -> [bz, num_labels]
            idx1 = (
                torch.arange(batch_size, device=logits.device)
                .reshape((-1, 1))
                .expand((-1, self.num_labels))
                .contiguous()
            )
            # [num_labels] -> [1, num_labels] -> [bz, num_labels] & [bz, 1] -> [bz, num_labels]
            idx2 = torch.arange(self.num_labels, device=logits.device).reshape(
                (1, -1)
            ).expand((batch_size, -1)).contiguous() + cls_idx.reshape((-1, 1))
            local_label_embeddings = hidden_states[idx1, idx2]  # [bz, num_labels, dim]
            # [bz, seq, dim] & [bz, dim, num_labels] -> [bz, seq, num_labels]
            logits = (
                torch.matmul(hidden_states, local_label_embeddings.transpose(-2, -1))
                * inv_temperature
            )
            # above to mat-multiply local_label_embeddings
        if self.config.loss_type in {"token_ce", "token_ce_intra"}:
            # below `pooled_logits` for output of evaluation only
            pooled_logits = logits
        return logits, pooled_logits

    def calculate_task_loss(
        self,
        task_labels,
        logits,
        pooled_logits,
        sample_wgt,
    ):
        labels = task_labels.to(logits.device)
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (
                labels.dtype == torch.long or labels.dtype == torch.int
            ):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        if self.config.problem_type == "regression":
            if self.config.loss_type == "l1":
                loss_fct = L1Loss()
                # reduction="sum"
                # refer to: https://github.com/microsoft/Graphormer/blob/main/graphormer/criterions/l1_loss.py#L35C26-L35C41
            else:
                loss_fct = MSELoss()
            labels = labels.to(dtype=pooled_logits.dtype)
            if self.num_labels == 1:
                loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(pooled_logits, labels)
        elif self.config.problem_type == "single_label_classification":
            if self.config.loss_type in {"token_ce", "token_ce_intra"}:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels).float(), labels.view(-1)
                )
            elif self.config.loss_type == "auc":
                # [batch, num_labels]
                logits = pooled_logits.view(-1, self.num_labels)
                y_pred = logits[:, 1].float() - logits[:, 0].float()  # [batch]
                loss = auc_loss(y_pred, labels.view(-1), self.config.num_neg)
            else:
                if sample_wgt is None:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        pooled_logits.view(-1, self.num_labels).float(),
                        labels.view(-1),
                    )
                else:
                    loss_fct = CrossEntropyLoss(reduction="none")
                    loss = loss_fct(
                        pooled_logits.view(-1, self.num_labels).float(),
                        labels.view(-1),
                    )
                    assert (
                        loss.shape[0] == sample_wgt.shape[0]
                    ), f"{loss.shape[0]} != {sample_wgt.shape[0]}"
                    loss = (
                        loss.float().view(-1) * sample_wgt.float().view(-1)
                    ).sum() / sample_wgt.float().sum()
        elif self.config.problem_type == "multi_label_classification":
            is_labeled = labels == labels
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = loss_fct(pooled_logits[is_labeled], labels[is_labeled])
            # remove `.float()` to avoid force-converting fp16 to fp32
            # labels[is_labeled] will convert tensor `labels` from 2D to 1D
        task_loss = loss
        return task_loss, logits, pooled_logits

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_raw_embeds: Optional[torch.FloatTensor] = None,
        task_labels: Optional[torch.LongTensor] = None,
        cls_idx: Optional[torch.LongTensor] = None,  # [bz]
        sample_wgt: Optional[torch.FloatTensor] = None,  # [bz], weight of sample
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, DoubleHeadsModelOutput]:
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

        if len(input_ids.size()) == 3:
            input_ids = input_ids[:, :, : self.config.stacked_feat]
        input_ids, inputs_embeds, in_ = self.prepare_inputs_embeds(
            input_ids, inputs_embeds, inputs_raw_embeds=inputs_raw_embeds
        )

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
        hidden_states = outputs[0]  # [N, seq, dim]

        # 1.1 Calculate logits for task, refer to `LlamaForSequenceClassification`
        logits = self.score(hidden_states)  # [N, seq, num_labels]

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        assert self.config.pad_token_id is not None
        sequence_lengths = _get_sequence_len(
            self.config.pad_token_id, in_, logits.device
        )
        assert self.pooling_method == "last", f"{self.pooling_method}!='last'"
        # [N, seq, num_labels] -> [N, num_labels]
        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]
        # 1.2 Calculate logits for special tasks, e.g., token-lvl task
        logits, pooled_logits = self.get_logits_for_token_lvl_task(
            batch_size, hidden_states, logits, pooled_logits, cls_idx
        )
        # 1.3 Calculate loss for task
        task_loss = None
        if task_labels is not None:
            task_loss, logits, pooled_logits = self.calculate_task_loss(
                task_labels, logits, pooled_logits, sample_wgt
            )
        # return results
        if not return_dict:
            output = (
                None,
                task_loss,
                None,
                pooled_logits.float(),
            ) + outputs[1:]
            return tuple(ele for ele in output if ele is not None)

        return DoubleHeadsModelOutput(
            pretrain_loss=None,
            task_loss=task_loss,
            pretrain_logits=None,
            task_logits=pooled_logits.float(),
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )


class GraphGPTDoubleHeadsModel(GraphGPTTaskModel):
    """
    Refer to `GPT2DoubleHeadsModel` in transformers/models/gpt2/modeling_gpt2.py
    Merge two models `LlamaForCausalLM` & `LlamaForSequenceClassification` in transformers/models/llama/modeling_llama.py
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # Init for CausalLM, refer to `LlamaForCausalLM`
        self.vocab_size = config.vocab_size
        if self.config.use_aux:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_raw_embeds: Optional[torch.FloatTensor] = None,
        pretrain_labels: Optional[torch.LongTensor] = None,
        task_labels: Optional[torch.LongTensor] = None,
        cls_idx: Optional[torch.LongTensor] = None,
        sample_wgt: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DoubleHeadsModelOutput]:
        r"""
        Args:
            pretrain_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            task_labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            cls_idx (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            sample_wgt (`torch.FloatTensor` of shape `(batch_size,)`, *optional*): weight of sample
        Returns:

        Example:

        ```python
        test
        ```"""
        res = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            inputs_raw_embeds=inputs_raw_embeds,
            task_labels=task_labels,
            cls_idx=cls_idx,
            sample_wgt=sample_wgt,
        )

        hidden_states = res.hidden_states  # [N, seq, dim]

        # Calculate loss for pre-train, refer to `LlamaForCausalLM`
        pretrain_loss = None
        pretrain_logits = None
        if self.config.use_aux:
            pretrain_logits = self.lm_head(hidden_states)
            logits = pretrain_logits.float()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, self.config.vocab_size)
            labels = pretrain_labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            pretrain_loss = loss_fct(logits, labels)

        return DoubleHeadsModelOutput(
            pretrain_loss=pretrain_loss,
            task_loss=res.task_loss,
            pretrain_logits=pretrain_logits,
            task_logits=res.task_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class GraphGPTDenoisingRegressionDoubleHeadsModel(GraphGPTTaskModel):
    """
    Refer to `GPT2DoubleHeadsModel` in transformers/models/gpt2/modeling_gpt2.py
    One Head for Supervised Regression task
    One Head for Denoising 3d coordinate Regression
    refer to https://aliyuque.antfin.com/james.zqf/ssqcu1/dexa1q0g8givelio?singleDoc# for details
    """

    def __init__(self, config):
        super().__init__(config)
        # 1. Init for Denoising Head
        # self.denoise = nn.Linear(config.hidden_size, 3, bias=False)
        self.denoise = utils_graphgpt.AtomTaskHead(config)
        self.noise_scale = 0.35
        self.denoise_wgt = 1
        self.denoise_schedule_pow = 0
        self.bi_causal = False
        r_2d, r_3d, r_both = 4.0, 0.0, 6.0  # 2D:3D:2D&3D
        self.mask_3d_ratio = r_2d / (r_2d + r_3d + r_both)
        self.mask_2d_ratio = r_3d / (r_3d + r_both)
        self.add_pos_type = True
        if self.bi_causal:
            self.model.bi_causal = self.bi_causal
            self.scale = MOL_ENERGY_SCALE
            bin_label_unit = (
                torch.tensor(
                    (2 ** torch.arange(MOL_ENERGY_BIN_LEN) / self.scale).tolist()[::-1]
                )
                .view((1, -1))
                .float()
            )
            self.register_buffer("bin_label_unit", bin_label_unit, persistent=False)
            # bin_label_wgt = torch.tensor([[1]], dtype=torch.float32)
            bin_label_wgt = (
                torch.tensor(
                    (
                        (torch.arange(MOL_ENERGY_BIN_LEN) + 1)
                        * 2
                        / (MOL_ENERGY_BIN_LEN + 1)
                    ).tolist()[::-1]
                )
                .view((1, -1))
                .float()
            )
            self.register_buffer("bin_label_wgt", bin_label_wgt, persistent=False)
        if self.add_pos_type:
            # embed 3D position type, 0~4: 0->pad, 1->(0,0,0), 2->(0,0,z), 3->(0,y,z), 4->(x,y,z)
            self.embed_pos_type = nn.Embedding(5, config.hidden_size, padding_idx=0)
        # 2. Use various transformation on 3D coordinates
        self.inputs_transform = "token-line"  # zero|raw|token-line|token-cube|token-mix
        self.num_bins = self.config.pos_bins
        assert self.inputs_transform in (
            "token-line",
            "token-cube",
            "token-mix",
        )
        if self.inputs_transform == "token-line":
            self.pos_agg_method = self.config.pos_agg_method  # sum|gated
            self._init_line_token_transform()
        elif self.inputs_transform == "token-cube":
            self._init_cube_token_transform()
        elif self.inputs_transform == "token-mix":
            self.num_bins_line = 256
            self.pos_agg_method = self.config.pos_agg_method  # sum|gated
            self.num_bins_cube = 32
            self._init_mix_token_transform()
        # 2.5 pos-bins config
        if hasattr(self.config, "pos_range"):
            pos_range = self.config.pos_range
        else:
            pos_range = "1p"
        range_min, range_max = DICT_range[pos_range]
        print(f"pos_range: {pos_range} -> {(range_min, range_max)}")
        self.register_buffer("range_min", range_min, persistent=False)
        self.register_buffer("range_max", range_max, persistent=False)
        # 2.7 raw-pos input projection
        self.use_pos_proj = False
        if self.use_pos_proj:
            self.in_pos_layernorm = modeling_llama.LlamaRMSNorm(
                3, eps=config.rms_norm_eps
            )
            self.in_pos_proj = nn.Linear(3, config.hidden_size, bias=False)
        # 3. Add additional aux-loss: pos-SMTP
        self.smtp_3d = False
        if self.smtp_3d:
            self.smtp_wgt = 1
            self.smtp_3d_scheduler_power = 0.1
            # `smtp_denoise == True` means un-masked noisy coord will be predicted using clean coord as the target!
            self.smtp_denoise = True
            self.smtp_vocab = 256
            self.smtp_proj = nn.Linear(
                config.hidden_size, 3 * config.hidden_size, bias=False
            )
            self.smtp_head = nn.Linear(config.hidden_size, self.smtp_vocab, bias=False)
        # randomly select samples + non-3d-info samples:: mask their node/edge attrs for training, not for inferring
        self.smtp_2d_rate = 0
        self.smtp_2d_scheduler_power = 0
        # Initialize weights and apply final processing
        self.post_init()

    def _init_line_token_transform(self):
        # discretize the (noisy) 3D position inputs
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
        if self.config.embed_dim > 0:
            del self.raw_embed_dropout
            del self.embed_layernorm
            del self.embed_proj

    def _init_cube_token_transform(self):
        # discretize the (noisy) 3D position inputs
        pos_vocab = self.num_bins**3 + 2
        self.embed_pos_token = nn.Embedding(
            pos_vocab, self.config.hidden_size, padding_idx=0
        )  # [pad]->0, [mask]->1
        if self.config.embed_dim > 0:
            del self.raw_embed_dropout
            del self.embed_layernorm
            del self.embed_proj
        self.register_buffer(
            "idx_multiplier",
            torch.tensor(
                [[self.num_bins**2, self.num_bins**1, self.num_bins**0]],
                dtype=torch.int64,
            ),
            persistent=False,
        )

    def _init_mix_token_transform(self):
        if self.config.embed_dim > 0:
            del self.raw_embed_dropout
            del self.embed_layernorm
            del self.embed_proj
        # discretize the (noisy) 3D position inputs

        # a). convert pos to line-tokens
        if self.pos_agg_method == "sum":
            pos_token_shift = torch.tensor(
                [[[0, self.num_bins_line, self.num_bins_line * 2]]], dtype=torch.int64
            )
            pos_vocab = self.num_bins_line * 3 + 2
        else:
            pos_token_shift = torch.tensor([[[0, 0, 0]]], dtype=torch.int64)
            pos_vocab = self.num_bins_line + 2
        self.register_buffer("pos_token_shift", pos_token_shift, persistent=False)
        self.embed_line_token = nn.Embedding(
            pos_vocab, self.config.hidden_size, padding_idx=0
        )  # [pad]->0, [mask]->1
        tmp_config = deepcopy(self.config)
        tmp_config.stacked_feat = 3
        tmp_config.stacked_feat_agg_method = self.pos_agg_method  # sum|gated
        self.line_token_agg = StackedFeatAggregation(tmp_config)

        # b). convert pos to cube-tokens
        pos_vocab = self.num_bins_cube**3 + 2
        self.embed_cube_token = nn.Embedding(
            pos_vocab, self.config.hidden_size, padding_idx=0
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

    def get_muon_params(self):
        # https://github.com/KellerJordan/Muon?tab=readme-ov-file#usage
        # Find ≥2D parameters in the body of the network -- these will be optimized by Muon
        muon_params = [p for p in self.model.layers.parameters() if p.ndim >= 2]
        # Find everything else -- these will be optimized by AdamW
        adamw_params = [p for p in self.model.layers.parameters() if p.ndim < 2]
        adamw_params.extend(self.model.norm.parameters())
        adamw_params.extend(self.model.embed_tokens.parameters())
        adamw_params.extend(self.stacked_feat_agg.parameters())
        adamw_params.extend(self.score.parameters())
        adamw_params.extend(self.denoise.parameters())
        adamw_params.extend(self.embed_pos_type.parameters())
        adamw_params.extend(self.embed_pos_token.parameters())
        adamw_params.extend(self.pos_token_agg.parameters())
        num_muon_params = 0
        for param in muon_params:
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
            if param.requires_grad:
                num_muon_params += num_params
        num_adamw_params = 0
        for param in adamw_params:
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
            if param.requires_grad:
                num_adamw_params += num_params
        tot_params = num_muon_params + num_adamw_params
        print(
            f"muon params: {num_muon_params}, adamw params: {num_adamw_params}, total params: {tot_params}"
        )
        return muon_params, adamw_params

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_raw_embeds: Optional[torch.FloatTensor] = None,
        *,
        pos_type: Optional[torch.Tensor] = None,  # [bz,seq]
        mask_per_token: Optional[torch.Tensor] = None,  # [bz,seq,1]
        **kwargs,
    ):
        input_ids, inputs_embeds, in_ = _get_stacked_inputs_embeds(
            self, input_ids, inputs_embeds
        )
        if self.add_pos_type:
            inputs_embeds += _get_pos_type_embeds(self.embed_pos_type, pos_type)

        if self.inputs_transform == "token-line":
            inputs_raw_embeds = transform_input_pos_via_line_token(
                self, inputs_raw_embeds, pos_type, mask_per_token
            )
        elif self.inputs_transform == "token-cube":
            inputs_raw_embeds = transform_input_pos_via_cube_token(
                self, inputs_raw_embeds, pos_type
            )
        elif self.inputs_transform == "token-mix":
            inputs_raw_embeds = transform_input_pos_via_mix_token(
                self, inputs_raw_embeds, pos_type
            )
        if inputs_raw_embeds is not None:
            inputs_embeds += inputs_raw_embeds
        return input_ids, inputs_embeds, in_

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_raw_embeds: Optional[torch.FloatTensor] = None,
        pretrain_labels: Optional[torch.LongTensor] = None,
        task_labels: Optional[torch.LongTensor] = None,
        cls_idx: Optional[torch.LongTensor] = None,
        sample_wgt: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DoubleHeadsModelOutput]:
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
        # 0. pre-process
        # 0.1 pre-process input_ids
        pos_deco = input_ids[:, :, self.config.stacked_feat :]  # [bz, seq, 3 or 4]
        pos_type = pos_deco[:, :, 0]  # [bz, seq] of 0/1/2/3/4
        node_idx = pos_deco[:, :, 2]
        input_ids = input_ids[:, :, : self.config.stacked_feat]
        raw_pos = inputs_raw_embeds[:, :, :3].contiguous()  # [bz, seq, 3]
        # 0.2 pre-process 3d positions inputs
        if self.training:
            mask_pos_ratio = self.mask_3d_ratio
        else:
            mask_pos_ratio = 1
        raw_pos = utils_graphgpt.apply_sample_lvl_mask_pos(raw_pos, mask_pos_ratio)

        # 0.25 2D-SMTP: pre-process 2d node/edge attrs inputs: applying 2D-SMTP mask on tokens
        if self.training:
            smtp_2d_rate = self.smtp_2d_rate
            power = self.smtp_2d_scheduler_power
        else:
            smtp_2d_rate = 0
            power = 0
        if smtp_2d_rate > 0 and power > 0:
            input_ids, _ = prepare_for_2d_smtp_inputs_labels(
                input_ids, node_idx, pos=raw_pos, smtp_2d_rate=smtp_2d_rate, power=power
            )

        # 0.3 deal with noise
        (
            clean_pos,
            noisy_pos,
            sample_mask,
            pad_mask,
            noise,
            noise_mask,
        ) = _add_pos_noise_and_get_masks(
            pos=raw_pos,
            pos_type=pos_type,
            noise_scale=self.noise_scale,
            node_idx=node_idx,
        )  # noise_mask: [bz, seq, 1]
        # print(f"[DEBUG] noise:\n{noise}\nnoise_mask:\n{noise_mask}")

        # 0.4 3D-SMTP: mask positions in node-lvl, polynomial scheduler
        mask_per_token = None
        if self.smtp_3d:
            (
                mask_per_token,
                noise,
                noisy_pos,
                noise_mask,
            ) = _mask_pos_in_node_lvl_on_schedule(
                noise,
                noisy_pos,
                noise_mask,
                pad_mask,
                node_idx,
                self.smtp_3d_scheduler_power,
            )

            # 0.5 create labels for above masked pos, for SMTP aux-loss
            smtp_labels = _get_labels_for_line_token(
                clean_pos,
                num_bins=self.smtp_vocab,
                range_min=self.range_min,
                range_max=self.range_max,
                sample_mask=sample_mask,
                pad_mask=pad_mask,
                mask_per_token=mask_per_token,
                pos_token_shift=self.pos_token_shift,
                denoise=self.smtp_denoise,
            )
            # print(f"[DEBUG] clean_pos:\n{clean_pos}\nsmtp_labels:\n{smtp_labels}")
        if self.denoise_schedule_pow != 0:
            assert self.smtp_3d is False
            (
                mask_per_token,
                noise,
                noisy_pos,
                noise_mask,
            ) = _mask_pos_in_node_lvl_on_schedule(
                noise,
                noisy_pos,
                noise_mask,
                pad_mask,
                node_idx,
                self.denoise_schedule_pow,
            )

        # obtain pair-wise 3D displacement
        # [bsz, seq, seq, 3]
        delta_pos, _ = utils_graphgpt.get_delta_pos(noisy_pos)

        # 0.6 deal with input-ids: transform to embeddings
        if self.training and self.mask_2d_ratio > 0:
            input_ids = utils_graphgpt.mask_2d(
                input_ids, noise_mask, self.mask_2d_ratio
            )
        input_ids, inputs_embeds, in_ = self.prepare_inputs_embeds(
            input_ids,
            inputs_embeds,
            noisy_pos,
            pos_type=pos_type,
            mask_per_token=mask_per_token,
        )

        if self.use_pos_proj:
            noisy_pos_embeds = self.in_pos_layernorm(noisy_pos.to(inputs_embeds.dtype))
            noisy_pos_embeds = self.in_pos_proj(noisy_pos_embeds)
            if self.embed_dropout is not None:
                noisy_pos_embeds = self.embed_dropout(noisy_pos_embeds)
            inputs_embeds += noisy_pos_embeds

        # 1. run backbone transformer
        # position_ids = None
        # print(f"[DEBUG] position_ids: {position_ids}")
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

        # 1. Calculate loss for supervised regression task, refer to `LlamaForSequenceClassification`
        if self.bi_causal:
            logits, labels, _ = _prepare_for_logits_labels_per_seq_lvl(
                raw_hidden_states, pretrain_labels, proj=self.score
            )
            logits = logits.view((-1, MOL_ENERGY_BIN_LEN))
            labels = labels.view((-1, MOL_ENERGY_BIN_LEN))
            # convert binary repr to decimal repr
            bin2decimal = (logits > 0).float() * self.bin_label_unit.float()
            pooled_logits = bin2decimal.sum(dim=-1)
        else:
            logits = self.score(raw_hidden_states)  # [bsz, seq, num_labels]
            batch_size = _get_batch_size(input_ids, inputs_embeds)
            assert self.config.pad_token_id is not None
            sequence_lengths = _get_sequence_len(
                self.config.pad_token_id, in_, logits.device
            )
            assert self.pooling_method == "last", f"{self.pooling_method}!='last'"
            # [N, seq, num_labels] -> [N, num_labels]
            bz_idx = torch.arange(batch_size, device=logits.device)
            pooled_logits = logits[bz_idx, sequence_lengths]

        task_loss = None
        if task_labels is not None:
            if self.bi_causal:
                loss_fct = nn.BCEWithLogitsLoss(reduction="none")

                task_loss = loss_fct(logits.float(), labels.float())
                # print(f"[DEBUG] logits:\n{logits}\nlabels:\n{labels}\ntask_loss:\n{task_loss}")
                task_loss = task_loss * self.bin_label_wgt.float()
                task_loss = task_loss.mean()
            else:
                loss_fct = L1Loss(reduction="none")
                task_labels = task_labels.to(pooled_logits)
                task_loss = loss_fct(pooled_logits.squeeze(), task_labels.squeeze())
                if sample_wgt is not None:
                    task_loss = task_loss * sample_wgt  # [bsz]
                task_loss = task_loss.mean()

        aux_loss = None
        if task_labels is not None and self.smtp_3d:
            dim = raw_hidden_states.shape[-1]
            (
                hidden_states,
                smtp_labels,
                wgt_3d,
            ) = _prepare_for_stacked_feat_labels_per_feat_lvl(
                raw_hidden_states,
                smtp_labels,
                proj=self.smtp_proj,
                num_feat=3,
            )
            hidden_states = hidden_states.view((-1, dim))
            logits = self.smtp_head(hidden_states)
            vocab_size = self.smtp_vocab
            aux_loss = _get_ce_loss(logits, smtp_labels, vocab_size)
            aux_loss = aux_loss * self.smtp_wgt

        # 2. Calculate loss for denoising task
        denoise_loss = None
        if pretrain_labels is not None:
            # node_output = logits = self.denoise(hidden_states)  # [bsz, seq, 3]
            logits = self.denoise(
                hidden_states=raw_hidden_states,
                delta_pos=delta_pos,
                position_ids=position_ids,
            )
            node_output_loss = utils_graphgpt.get_denoise_loss(
                logits, noise_mask, noise
            )
            denoise_loss = node_output_loss * self.denoise_wgt
        if denoise_loss is not None and aux_loss is not None:
            denoise_loss += aux_loss
        return DoubleHeadsModelOutput(
            pretrain_loss=denoise_loss,
            task_loss=task_loss,
            pretrain_logits=logits,
            task_logits=pooled_logits.float(),
        )


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


def _update_causal_mask(self, attention_mask, input_tensor, **kwargs):
    # print(attention_mask)
    if hasattr(self, "bi_causal") and self.bi_causal:
        # print(_prepare_4d_bi_causal_attention_mask(attention_mask, input_tensor.dtype))
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
            # print(f"nonzero_feat: {nonzero_feat}\nratio: {ratio}")
    else:
        in_ = input_ids  # [bz, seq]
    input_ids = None
    return input_ids, inputs_embeds, in_


def _get_ce_loss(
    logits,
    labels,
    vocab_size,
    *,
    wgt=None,
    label_smoothing: float = 0,
    focal_gamma: float = 0,
):
    # print(f"[DEBUG] wgt: {wgt}, focal_gamma: {focal_gamma}")
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


def _get_cl_logits_loss(
    cl_proj: torch.nn.Module,
    raw_hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
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


def _prepare_for_stacked_feat_labels_per_feat_lvl(
    hidden_states: torch.FloatTensor,
    labels: torch.LongTensor,
    *,
    proj: torch.nn.Module,
    num_feat: int,
):
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
    # iv). obtain normalized wgt
    wgt = mask_m.float()  # [N, seq, n_token]
    wgt = wgt / (wgt.sum(dim=-1).sum(dim=-1)[:, None, None] + _EPSILON)
    # [N, seq, n_token] -> [M]
    wgt = wgt[mask_m]
    return hidden_states, labels, wgt


def _prepare_for_logits_labels_per_seq_lvl(
    hidden_states: torch.FloatTensor,
    labels: torch.LongTensor,
    *,
    proj: torch.nn.Module,
):
    # i). obtain mask
    mask = labels != -100  # [N, seq]
    # ii). deal hidden states: mask and reshape
    # [N, seq, dim] -> [M, dim]
    hidden_states = hidden_states[mask]
    # [M, dim] -> [M, dim]
    hidden_states = proj(hidden_states)
    # iii). deal labels: mask and reshape
    # [N, seq] -> [M]
    labels = labels[mask]
    # iv). obtain normalized wgt
    wgt = mask.float()  # [N, seq]
    wgt = wgt / (wgt.sum(dim=-1)[:, None] + _EPSILON)
    # [N, seq, n_token] -> [M]
    wgt = wgt[mask]
    return hidden_states, labels, wgt


def prepare_for_stacked_feat_labels(
    self,
    hidden_states: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
):
    if self.config.stack_method == "long":
        hidden_states, labels, wgt = _prepare_for_stacked_feat_labels_per_feat_lvl(
            hidden_states,
            labels,
            proj=self.n_token_proj,
            num_feat=self.config.next_n_token,
        )
    # the version below can save lots of GPU memory and boost speed
    if self.config.stack_method == "short":
        dim = hidden_states.shape[-1]  # [N, seq, dim]
        # i). obtain mask
        if labels is not None:
            labels_m = labels[:, :, 0]  # [N, seq, next_n] -> [N, seq]
            mask_m = labels_m != -100  # [N, seq]
        else:  # for inference
            mask_m = torch.ones(hidden_states.size()[:2], dtype=torch.bool)
        # ii). deal hidden states: mask and reshape
        # [N, seq, dim] -> [M, dim]
        hidden_states = hidden_states[mask_m]
        # [M, dim] -> [M, dim*next_n]
        hidden_states = self.n_token_proj(hidden_states)
        # [M, dim*next_n] -> [M*next_n, dim]
        hidden_states = hidden_states.reshape((-1, dim))
        # iii). deal labels: mask and reshape
        if labels is not None:
            # [N, seq, next_n] -> [M, next_n]
            labels = labels[mask_m]
            # [M, next_n] -> [M*next_n]
            labels = labels.reshape(-1)
    return hidden_states, labels


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
    # print(f"[DEBUG] Invoking `prepare_for_2d_smtp_inputs`. input_ids:\n{input_ids}\npos:\n{pos}\nnode_idx:\n{node_idx}")
    # print(f"[DEBUG] smtp_2d_rate: {smtp_2d_rate}, power: {power}")

    # 1. get sample-lvl mask for samples for 2D-smtp
    sample_mask = (
        torch.rand((bz, 1, 1), dtype=torch.float32, device=device) < smtp_2d_rate
    )
    # re-calculate sample-mask, because some mols doesn't has pos, i.e., pos all 0's
    if pos is not None:
        sample_mask_pos = (pos.abs() < eps).all(dim=-1).all(dim=-1)
        sample_mask = sample_mask_pos[:, None, None] | sample_mask
    # print(f"[DEBUG] sample_mask:\n{sample_mask}")
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
    # print(f"[DEBUG] mask_per_node:\n{mask_per_node}")
    mask_per_token = mask_per_node[bz_idx, node_idx]  # look-up
    # print(f"[DEBUG] mask_per_token:\n{mask_per_token}")
    mask_per_token = mask_per_token & (input_ids > 0)
    # print(f"[DEBUG] mask_per_token:\n{mask_per_token}")
    labels = input_ids.clone().masked_fill_(~mask_per_token, label_pad_token_id)
    # print(f"[DEBUG] labels:\n{labels}")
    input_ids = input_ids.clone().masked_fill_(mask_per_token, mask_token_id)
    # print(f"[DEBUG] input_ids:\n{input_ids}")
    # 3. For [mask] tokens, randomly replace some with `token draw from vocab`
    # rnd_tokens = _get_uniform_rnd_tokens(raw_input_ids, vocab)
    rnd_tokens = _get_gaussian_rnd_tokens(raw_input_ids, vocab)
    # print(f"[DEBUG] rnd_tokens:\n{rnd_tokens}")
    replace_mask = (
        torch.rand((bz, seq, feat), dtype=torch.float32, device=device) < replace_rate
    )
    # print(f"[DEBUG] replace_mask:\n{replace_mask}")
    replace_mask = mask_per_token & replace_mask
    # print(f"[DEBUG] replace_mask:\n{replace_mask}")
    input_ids = input_ids * (~replace_mask).long() + rnd_tokens * replace_mask.long()
    # print(f"[DEBUG] input_ids:\n{input_ids}")
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
    # print(f"[DEBUG] mask:\n{mask}")
    # [bz, seq, 3] -> [bz, seq] -> [bz]
    sample_mask = mask.all(dim=-1).all(dim=-1, keepdim=True)  # [bz, 1]
    # print(f"[DEBUG] sample_mask:\n{sample_mask}")
    return sample_mask


def _mask_pos_in_node_lvl_on_schedule(
    noise, noisy_pos, noise_mask, pad_mask, node_idx, power
):
    # print(f"[DEBUG] INIT, noise:\n{noise}\nnoisy_pos:\n{noisy_pos}\nnoise_mask:\n{noise_mask}")
    mask_per_node, mask_per_coord, bz_idx = _preprocess_pos_smtp_masks(
        noisy_pos, power=power
    )  # power==0 <=> no 3d-SMTP
    # print(f"[DEBUG] mask_per_node:\n{mask_per_node}")
    mask_per_token = _get_mask_per_token_for_line(
        mask_per_node, mask_per_coord, pad_mask, bz_idx, node_idx, False
    )  # [bz, seq, 3]
    mask_per_token = mask_per_token[:, :, 0:1]
    # print(f"[DEBUG] mask_per_token:\n{mask_per_token}")
    noise = noise.masked_fill_(mask_per_token, 0)
    noisy_pos = noisy_pos.masked_fill_(mask_per_token, 0)
    noise_mask = noise_mask | mask_per_token
    # print(f"[DEBUG] AFTER mask_per_token, noise:\n{noise}\nnoisy_pos:\n{noisy_pos}\nnoise_mask:\n{noise_mask}")
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
    # print(f"[DEBUG] 0. pos_tokens:\n{pos_tokens}")
    # 1. fill the sample lvl: [mask]!
    # IF [pad], slightly worse results => for transfer learning where mols' pos it not available
    pos_tokens = pos_tokens.masked_fill_(sample_mask[:, :, None], mask_token_id)
    # print(f"[DEBUG] 1. pos_tokens:\n{pos_tokens}")
    # 2. fill token-coordinate-lvl masked positions with [mask]
    if mask_per_token is not None:
        pos_tokens = pos_tokens.masked_fill_(mask_per_token, mask_token_id)
        # print(f"[DEBUG] 2. pos_tokens:\n{pos_tokens}")
    # 3. fill the token lvl: [pad]
    pos_tokens = pos_tokens.masked_fill_(~pad_mask[:, :, None], pad_token_id)
    # print(f"[DEBUG] 3. pad_mask:\n{pad_mask}\npos_tokens:\n{pos_tokens}")
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
    # print(f"[DEBUG] pos_tokens:\n{pos_tokens}")
    # 1.32 fill the token lvl: [pad]
    # print(f"[DEBUG] pad_mask:\n{pad_mask}")
    pos_tokens = pos_tokens.masked_fill_(~pad_mask, pad_token_id)
    # print(f"[DEBUG] pos_tokens:\n{pos_tokens}")
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
    # print(f"[DEBUG] Invoking `transform_pos_via_3d_token`")
    # print(f"[DEBUG] pos:\n{pos}\npos_tokens:\n{pos_tokens}")
    # [bz, seq, 3] & [1, 3] -> [bz, seq, 3] -> [bz, seq]
    pos_tokens = (pos_tokens * self.idx_multiplier[None, :, :]).sum(dim=-1) + 2
    # print(f"[DEBUG] pos_tokens:\n{pos_tokens}\npos_tokens.shape:\n{pos_tokens.shape}")

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
    # print(f"[DEBUG] pos:\n{pos}\npos_type:{pos_type}")
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
    # print(f"[DEBUG] line-tokens:\n{pos_tokens}")
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
    # print(f"[DEBUG] cube-tokens:\n{pos_tokens}")
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
    # print(f"[DEBUG] invoking prepare_pos_smtp_line_token_inputs_and_labels.\npos:{pos}\npos_type:\n{pos_type}\nnode_idx:\n{node_idx}")
    # print(f"[DEBUG] invoking prepare_pos_smtp_line_token_inputs_and_labels.\nrange_min:{self.range_min}\nrange_max:{self.range_max}\nnum_bins:{self.num_bins}")

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
    # print(f"[DEBUG] pos_embeds:\n{pos_embeds}\npos_embeds.shape: {pos_embeds.shape}")
    if self.embed_dropout is not None:
        pos_embeds = self.embed_dropout(pos_embeds)
    pos_embeds = self.pos_token_agg(pos_embeds)  # [bz, seq, dim]
    # print(f"[DEBUG] pos_embeds:\n{pos_embeds}\npos_embeds.shape: {pos_embeds.shape}")
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
    # print(f"[DEBUG] Invoking `_get_inputs_for_line_token`.\ninput_tokens:\n{input_tokens}")
    # print(f"[DEBUG] mask_per_token:\n{mask_per_token}")
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
    # print(f"[DEBUG] 1. labels:\n{labels}")
    # 2. fill token-lvl non-masked positions with [label-pad]
    if denoise:
        labels = labels.masked_fill_(~pad_mask[:, :, None], label_pad_token_id)
    else:
        labels = labels.masked_fill_(~mask_per_token, label_pad_token_id)
    # print(f"[DEBUG] 2. labels:\n{labels}")
    return labels


def prepare_pos_smtp_cube_token_inputs_and_labels(self, pos, pos_type, node_idx):
    # pos: [bz, seq, 3], pos_type: [bz, seq], node_idx: [bz, seq]
    noise_scale = self.smtp_3d_noise_scale  # set 0 to NOT DENOISE in SMTP pre-train
    # SMTP scheduler
    power = self.smtp_3d_power
    gt_rate = 0  # rate to unveil [mask] node's coord
    # print(f"[DEBUG] invoking prepare_pos_smtp_cubic_token_inputs_and_labels.\npos:{pos}\npos_type:\n{pos_type}\nnode_idx:\n{node_idx}")
    # print(f"[DEBUG] range_min:{self.range_min}\nrange_max:{self.range_max}\nnum_bins:{self.num_bins}")

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
    # print(f"[DEBUG] pos_embeds:\n{pos_embeds}\npos_embeds.shape: {pos_embeds.shape}")
    if self.embed_dropout is not None:
        pos_embeds = self.embed_dropout(pos_embeds)
    # print(f"[DEBUG] pos_embeds:\n{pos_embeds}\npos_embeds.shape: {pos_embeds.shape}")
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
    # print(f"[DEBUG] input_tokens:\n{input_tokens}")
    # 3.1 deal with inputs pos tokens
    # 3.11 fill sample-lvl zeros' positions with [mask]
    input_tokens = (input_tokens * idx_multiplier[None, :, :]).sum(dim=-1) + 2
    input_tokens = input_tokens.masked_fill_(sample_mask, mask_token_id)
    # print(f"[DEBUG] 3.11 input_tokens:\n{input_tokens}")
    # 3.12 fill token-lvl masked positions with [mask]
    input_tokens = input_tokens.masked_fill_(mask_per_token, mask_token_id)
    # print(f"[DEBUG] 3.12 input_tokens:\n{input_tokens}")
    # 3.13 fill token-lvl padded positions wtih [pad]
    input_tokens = input_tokens.masked_fill_(~pad_mask, pad_token_id)
    # print(f"[DEBUG] 3.13 input_tokens:\n{input_tokens}")
    # 3.2 deal with labels of pos tokens from clean pos
    labels = discrete_pos(
        clean_pos,
        num_bins,
        dict_bounds=None,
        range_min=range_min,
        range_max=range_max,
    )  # [bz, seq, 3]
    # print(f"[DEBUG] 3.2 labels:\n{labels}")
    # 3.21 fill sample-lvl zeros' positions with [label-pad]
    labels = (labels * idx_multiplier[None, :, :]).sum(dim=-1) + 2
    labels = labels.masked_fill_(sample_mask, label_pad_token_id)
    # print(f"[DEBUG] 3.21 labels:\n{labels}")
    # 3.22 fill token-lvl non-masked positions with [label-pad]
    labels = labels.masked_fill_(~mask_per_token, label_pad_token_id)
    # print(f"[DEBUG] 3.22 labels:\n{labels}")
    return input_tokens, labels


def prepare_pos_smtp_mix_token_inputs_and_labels(
    self, pos, pos_type, node_idx, apply_denoise: bool = False
):
    # pos: [bz, seq, 3], pos_type: [bz, seq], node_idx: [bz, seq]
    noise_scale = self.smtp_3d_noise_scale  # set 0 to NOT DENOISE in SMTP pre-train
    # SMTP scheduler
    power = self.smtp_3d_power
    gt_rate = 0  # rate to unveil [mask] node's coord
    # print(f"[DEBUG] invoking prepare_pos_smtp_mix_token_inputs_and_labels.\npos:{pos}\npos_type:\n{pos_type}\nnode_idx:\n{node_idx}")
    # print(f"[DEBUG] range_min:{self.range_min}\nrange_max:{self.range_max}\nnum_bins_line:{self.num_bins_line}\nnum_bins_cube:{self.num_bins_cube}")

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
    # print(f"[DEBUG] mask_per_token_line:\n{mask_per_token_line}\nmask_per_token_line.shape: {mask_per_token_line.shape}")
    # b). For cube token
    mask_per_token_cube = _get_mask_per_token_for_cube(
        mask_per_node, mask_per_coord, pad_mask, bz_idx, node_idx
    )
    # print(f"[DEBUG] mask_per_token_cube:\n{mask_per_token_cube}\nmask_per_token_cube.shape: {mask_per_token_cube.shape}")

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
    # print(f"[DEBUG] line_embeds:\n{line_embeds}\nline_embeds.shape: {line_embeds.shape}")
    cube_embeds = self.embed_cube_token(input_tokens_cube)  # [bz, seq, dim]
    if self.embed_dropout is not None:
        line_embeds = self.embed_dropout(line_embeds)
        cube_embeds = self.embed_dropout(cube_embeds)
    line_embeds = self.line_token_agg(line_embeds)  # [bz, seq, dim]
    # print(f"[DEBUG] line_embeds:\n{line_embeds}\nline_embeds.shape: {line_embeds.shape}")
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
    # print(f"[DEBUG] power: {power}\nmr_per_sample:\n{mr_per_sample}")
    # apply polynomial mask on node-idx level
    mask_per_node = (
        torch.rand((bz, seq, 3), dtype=torch.float32, device=device) > mr_per_sample
    )
    # print(f"[DEBUG] mask_per_node:\n{mask_per_node}")
    mask_per_coord = (
        torch.rand((bz, seq, 3), dtype=torch.float32, device=device) > gt_rate
    )
    # print(f"[DEBUG] mask_per_coord:\n{mask_per_coord}")
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
    # print(f"[DEBUG] invoking `_add_pos_noise_and_get_masks`")
    # print(f"[DEBUG] noise_scale: {noise_scale}\npos_type:\n{pos_type}")
    # 1. get sample-level mask
    eps = _EPSILON
    mask = pos.abs() < eps  # [bz, seq, 3]
    # print(f"[DEBUG] mask:\n{mask}")
    pos = pos.masked_fill_(mask, 0)
    # print(f"[DEBUG] pos:\n{pos}\npos.shape: {pos.shape}")
    # 2. obtain noise-mask, new sample-mask directly from pos: will include mol with pos==0
    sample_mask = mask.all(dim=-1).all(dim=-1, keepdim=True)  # [bz, 1]
    if pos_type is not None:
        pad_mask = pos_type > 0  # [bz, seq]
        noise_mask = (~pad_mask) | sample_mask  # [bz, seq]
    else:
        noise_mask = mask.all(dim=-1)  # [bz, seq]
        pad_mask = ~noise_mask
    # print(f"[DEBUG] noise_mask:\n{noise_mask}")
    noise_mask = noise_mask[:, :, None]
    # 3. add noise to clean pos, and apply mask
    gnoise = torch.randn(pos.shape).to(pos) * noise_scale
    gnoise = gnoise[bz_idx, node_idx].contiguous()
    noise = gnoise.masked_fill_(noise_mask, 0.0)
    # print(f"[DEBUG] noise:\n{noise}\nnode_idx:\n{node_idx}")
    noisy_pos = pos + noise
    # print(f"[DEBUG] noisy_pos:\n{noisy_pos}\nnoisy_pos.shape: {noisy_pos.shape}")
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
    # print(f"[DEBUG] Invoking `_get_mask_per_token_for_line`.\ncoord_lvl_mask: {coord_lvl_mask}")
    if not coord_lvl_mask:
        mask_per_node = mask_per_node[:, :, 0:1]  # [bz, seq, 1]
    mask_per_node = mask_per_node & mask_per_coord
    # print(f"[DEBUG] mask_per_node:\n{mask_per_node}")
    # get mask on line token level through look-up
    mask_per_token = mask_per_node[bz_idx, node_idx]
    mask_per_token = mask_per_token & pad_mask[:, :, None]
    # print(f"[DEBUG] mask_per_node & pad_mask -> mask_per_node:\n{mask_per_node}")
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
    # print(f"[DEBUG] mask_per_token & pad_mask -> mask_per_token:\n{mask_per_token_cube}")
    return mask_per_token


def _get_pos_type_embeds(embed_lookup: torch.nn.Module, pos_type: torch.Tensor):
    # Converting 3D position type to embeddings
    # sample_mask = _get_sample_lvl_mask(pos)  # [bz, 1]
    # pos_type = pos_type.clone().masked_fill_(sample_mask, 0)
    # ABOVE 2 lines pad `pos_type` with 0 => will slightly worsen fine-tune results, so remove it!
    pos_type = torch.clamp(pos_type, min=0)  # [bz, seq]
    type_embeds = embed_lookup(pos_type)  # [bz, seq, dim]
    return type_embeds
