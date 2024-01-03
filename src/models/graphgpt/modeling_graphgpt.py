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
import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, L1Loss
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    ModelOutput,
)
from transformers.models.llama.modeling_llama import (
    LLAMA_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
    LLAMA_START_DOCSTRING,
)
from transformers import LlamaPreTrainedModel, LlamaForCausalLM
from transformers.models.llama import modeling_llama

from . import utils_graphgpt
from src.utils.modules_utils import MLP
from src.utils.loss_utils import auc_loss


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
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class GraphGPTCausal(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        if _use_dropout(config):
            del self.model
            self.model = utils_graphgpt.LlamaModel(config)

            # Initialize weights and apply final processing
            self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        prior: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits + (label_mask - 1.0) * 1e9 if label_mask is not None else logits
        if prior is not None:  # https://spaces.ac.cn/archives/7615
            logits = logits.float() + prior

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, self.config.vocab_size)
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits.float(), labels)
            # convert logits to float before cross-entropy for molecule datasets like PCQM4M-v2, MOLPCBA and etc.
            # because when batch-size too large, ce with fp16 leads to no decrease of loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class GraphGPTDoubleHeadsModel(LlamaPreTrainedModel):
    """
    Refer to `GPT2DoubleHeadsModel` in transformers/models/gpt2/modeling_gpt2.py
    Merge two models `LlamaForCausalLM` & `LlamaForSequenceClassification` in transformers/models/llama/modeling_llama.py
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        LlamaModel = (
            utils_graphgpt.LlamaModel
            if _use_dropout(self.config)
            else modeling_llama.LlamaModel
        )
        self.model = LlamaModel(config)
        # 1. Init for CausalLM, refer to `LlamaForCausalLM`
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 2. Init for SequenceClassification, refer to `LlamaForSequenceClassification`
        self.num_labels = config.num_labels
        self.score = MLP(
            config.hidden_size,
            self.num_labels,
            mlp=self.config.mlp,
            hidden_act=self.config.hidden_act,
            dropout=self.config.dropout,
        )
        self.pos_weight = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pretrain_labels: Optional[torch.LongTensor] = None,
        task_labels: Optional[torch.LongTensor] = None,
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
        Returns:

        Example:

        ```python
        test
        ```"""

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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        hidden_states = outputs[0]

        # 1. Calculate loss for pre-train, refer to `LlamaForCausalLM`
        pretrain_logits = self.lm_head(hidden_states)
        logits = pretrain_logits.float()

        pretrain_loss = None
        if pretrain_labels is not None:
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, self.config.vocab_size)
            labels = pretrain_labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            pretrain_loss = loss_fct(logits, labels)

        # 2. Calculate loss for task, refer to `LlamaForSequenceClassification`
        logits = self.score(hidden_states)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        task_loss = None
        if task_labels is not None:
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
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    task_loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    task_loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                task_loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                task_loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (
                pretrain_loss,
                task_loss,
                pretrain_logits,
                pooled_logits,
            ) + outputs[1:]
            return tuple(ele for ele in output if ele is not None)

        return DoubleHeadsModelOutput(
            pretrain_loss=pretrain_loss,
            task_loss=task_loss,
            pretrain_logits=pretrain_logits,
            task_logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GraphGPTUniBiDoubleHeadsModel(LlamaPreTrainedModel):
    """
    Refer to `GPT2DoubleHeadsModel` in transformers/models/gpt2/modeling_gpt2.py
    One Head for Causal NTP task -> Unidirectional attn
    One Head for Regression -> Bidirectional attn
    refer to https://aliyuque.antfin.com/james.zqf/ssqcu1/dexa1q0g8givelio?singleDoc# for details
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = utils_graphgpt.UniBiLlamaModel(config)
        # 1. Init for CausalLM, refer to `LlamaForCausalLM`
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 2. Init for SequenceClassification, refer to `LlamaForSequenceClassification`
        self.num_labels = config.num_labels
        if len(self.config.mlp) > 0:
            self.score = MLP(
                config.hidden_size,
                self.num_labels,
                mlp=self.config.mlp,
                hidden_act=self.config.hidden_act,
                dropout=self.config.dropout,
            )
        else:
            self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.pos_weight = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_bi: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pretrain_labels: Optional[torch.LongTensor] = None,
        task_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DoubleHeadsModelOutput]:
        r"""
        Args:
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                2D mask contains both uni- & bi- attn mask
            attention_mask_bi (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                2D mask contains ONLY bi- attn mask
            pretrain_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            task_labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:

        Example:

        ```python
        test
        ```"""

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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_mask_bi=attention_mask_bi,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # [bsz, seq, dim]

        # 1. Calculate loss for pre-train, refer to `LlamaForCausalLM`
        pretrain_logits = self.lm_head(hidden_states)  # [bsz, seq, vocab]
        logits = pretrain_logits.float()  # [bsz, seq, vocab]

        pretrain_loss = None
        if pretrain_labels is not None:
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, self.config.vocab_size)  # [bsz*seq, vocab]
            labels = pretrain_labels.view(-1)  # [bsz*seq]
            # Enable model parallelism
            labels = labels.to(logits.device)
            pretrain_loss = loss_fct(logits, labels)

        # 2. Calculate loss for task, refer to `LlamaForSequenceClassification`
        eps = 1e-7
        logits = self.score(hidden_states)  # [bsz, seq, num_labels]
        assert task_labels is not None
        labels = task_labels.to(dtype=logits.dtype, device=logits.device)
        assert (
            self.config.problem_type == "regression"
        ), f"problem_type {self.config.problem_type} NOT implemented"
        loss_fct = MSELoss(reduction="none")
        task_loss = loss_fct(logits, labels)  # [bsz, seq, num_labels]
        bi_mask = attention_mask_bi[:, :, None].to(task_loss.dtype)  # [bsz, seq, 1]
        task_loss = task_loss * bi_mask  # [bsz, seq, num_labels]
        task_loss = task_loss.sum() / (bi_mask.sum() + eps)

        if not return_dict:
            output = (
                pretrain_loss,
                task_loss,
                pretrain_logits,
                logits,
            ) + outputs[1:]
            return tuple(ele for ele in output if ele is not None)

        return DoubleHeadsModelOutput(
            pretrain_loss=pretrain_loss,
            task_loss=task_loss,
            pretrain_logits=pretrain_logits,
            task_logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class GraphGPTForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 1. transformer backbone
        LlamaModel = (
            utils_graphgpt.LlamaModel
            if _use_dropout(self.config)
            else modeling_llama.LlamaModel
        )
        self.model = LlamaModel(config)
        # 2. mlp for supervised fine-tune tasks
        self.num_labels = config.num_labels
        if len(self.config.mlp) > 0:
            self.score = MLP(
                config.hidden_size,
                self.num_labels,
                mlp=self.config.mlp,
                hidden_act=self.config.hidden_act,
                dropout=self.config.dropout,
            )
        else:
            self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.pos_weight = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]  # [batch, seq, dim]
        logits = self.score(hidden_states)  # [batch, seq, num_labels]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]  # [batch, num_labels]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
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
                    # reduction="sum" -> remove to recover lf's best result
                    # refer to: https://github.com/microsoft/Graphormer/blob/main/graphormer/criterions/l1_loss.py#L35C26-L35C41
                else:
                    loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                    # remove .float() to recover lf's best result!!!
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.config.loss_type == "auc":
                    logits = pooled_logits.view(
                        -1, self.num_labels
                    )  # [batch, num_labels]
                    y_pred = logits[:, 1].float() - logits[:, 0].float()  # [batch]
                    loss = auc_loss(y_pred, labels.view(-1), self.config.num_neg)
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        pooled_logits.view(-1, self.num_labels), labels.view(-1)
                    )
            elif self.config.problem_type == "multi_label_classification":
                is_labeled = labels == labels
                loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                loss = loss_fct(
                    pooled_logits[is_labeled], labels[is_labeled]
                )  # remove `.float()` to avoid force-converting fp16 to fp32
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


def _use_dropout(config):
    if (
        sum(
            [config.embd_pdrop, config.attn_pdrop, config.resid_pdrop, config.mlp_pdrop]
        )
        > 0
    ):
        print("Applying dropout in backbone transformer")
        return True
    else:
        print("NOT Applying dropout in backbone transformer")
        return False
