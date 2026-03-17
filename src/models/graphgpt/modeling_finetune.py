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
Fine-tuning model classes: GraphGPTTaskModel, GraphGPTDoubleHeadsModel,
GraphGPTDenoisingRegressionDoubleHeadsModel.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, L1Loss
from typing import List, Optional, Tuple, Union
from transformers import LlamaPreTrainedModel
from transformers.models.llama import modeling_llama

from . import utils_graphgpt
from .modeling_common import (
    DoubleHeadsModelOutput,
    StackedFeatAggregation,
    init_backbone,
    init_embed_dropout,
    init_stacked_feat_agg,
    resolve_forward_defaults,
)
from .modeling_helpers import (
    _update_causal_mask,
    _get_batch_size,
    _get_sequence_len,
    _get_stacked_inputs_embeds,
    _get_pos_type_embeds,
    _get_ce_loss,
    _get_labels_for_line_token,
    _prepare_for_logits_labels_per_seq_lvl,
    _prepare_for_stacked_feat_labels_per_feat_lvl,
    _add_pos_noise_and_get_masks,
    _mask_pos_in_node_lvl_on_schedule,
    transform_inputs_raw_embeds,
    transform_input_pos_via_line_token,
    transform_input_pos_via_cube_token,
    transform_input_pos_via_mix_token,
    prepare_for_2d_smtp_inputs_labels,
)
from src.utils.modules_utils import MLP
from src.utils.loss_utils import auc_loss
from src.utils.mol_utils import DICT_range
from src.utils.tokenizer_utils import MOL_ENERGY_BIN_LEN, MOL_ENERGY_SCALE


class GraphGPTTaskModel(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # 1. Transformer's backbone
        init_backbone(self, config)
        # 1.1 Embedding dropout
        init_embed_dropout(self, config)
        # 1.2 Node/edge attributes stacking
        init_stacked_feat_agg(self, config, conditional=True)
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
        output_attentions, output_hidden_states, return_dict, position_ids = (
            resolve_forward_defaults(
                self, output_attentions, output_hidden_states, return_dict, position_ids
            )
        )

        if len(input_ids.size()) == 3:
            input_ids = input_ids[:, :, : self.config.stacked_feat]
        input_ids, inputs_embeds, in_ = self.prepare_inputs_embeds(
            input_ids, inputs_embeds, inputs_raw_embeds=inputs_raw_embeds
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
        # [N, seq, dim] -> [N, dim]
        pooled_hidden_states = hidden_states[
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
            task_hidden_states=pooled_hidden_states,
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
        self.denoise = utils_graphgpt.AtomTaskHead(config)
        self.noise_scale = config.noise_scale
        self.denoise_wgt = config.denoise_wgt
        self.denoise_schedule_pow = config.denoise_schedule_pow
        self.bi_causal = config.bi_causal
        r_2d, r_3d, r_both = config.r_2d, config.r_3d, config.r_both
        self.mask_3d_ratio = r_2d / (r_2d + r_3d + r_both)
        self.mask_2d_ratio = r_3d / (r_3d + r_both)
        self.add_pos_type = config.add_pos_type
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
        self.inputs_transform = config.inputs_transform
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
            self.num_bins_line = config.num_bins_line
            self.pos_agg_method = self.config.pos_agg_method  # sum|gated
            self.num_bins_cube = config.num_bins_cube
            self._init_mix_token_transform()
        # 2.5 pos-bins config
        pos_range = config.dn_pos_range
        range_min, range_max = DICT_range[pos_range]
        print(f"pos_range: {pos_range} -> {(range_min, range_max)}")
        self.register_buffer("range_min", range_min, persistent=False)
        self.register_buffer("range_max", range_max, persistent=False)
        # 2.7 raw-pos input projection
        self.use_pos_proj = config.dn_use_pos_proj
        if self.use_pos_proj:
            self.in_pos_layernorm = modeling_llama.LlamaRMSNorm(
                3, eps=config.rms_norm_eps
            )
            self.in_pos_proj = nn.Linear(3, config.hidden_size, bias=False)
        # 3. Add additional aux-loss: pos-SMTP
        self.smtp_3d = config.smtp_3d
        if self.smtp_3d:
            self.smtp_wgt = config.smtp_wgt
            self.smtp_3d_scheduler_power = config.smtp_3d_scheduler_power
            self.smtp_denoise = config.smtp_denoise
            self.smtp_vocab = config.smtp_vocab
            self.smtp_proj = nn.Linear(
                config.hidden_size, 3 * config.hidden_size, bias=False
            )
            self.smtp_head = nn.Linear(config.hidden_size, self.smtp_vocab, bias=False)
        # randomly select samples + non-3d-info samples:: mask their node/edge attrs for training, not for inferring
        self.smtp_2d_rate = config.dn_smtp_2d_rate
        self.smtp_2d_scheduler_power = config.smtp_2d_scheduler_power
        # Initialize weights and apply final processing
        self.post_init()

    # NOTE: similar _init_*_token_transform logic exists in GraphGPTPosPred,
    # but with additional projection heads for prediction.
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
        output_attentions, output_hidden_states, return_dict, position_ids = (
            resolve_forward_defaults(
                self, output_attentions, output_hidden_states, return_dict, position_ids
            )
        )
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
