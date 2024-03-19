# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from typing import Any, Optional, List, Dict, Union, Tuple
from torch_geometric.data import Data
from .tokenizer import GSTTokenizer


@dataclass
class DataCollatorForGSTCausal:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: GSTTokenizer
    model: Optional[Any] = None
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    mask_boundary: bool = False
    global_steps: Optional[int] = None
    total_num_steps: Optional[int] = None
    num_workers: Optional[int] = None

    def __call__(
        self, graphs: List[Union[Tuple[int, Data], Dict]], return_tensors=None
    ):
        if return_tensors is None:
            return_tensors = self.return_tensors

        def _add_idx_to_dict(dict_: Dict, index: int):
            dict_.update({"idx": index})
            return dict_

        if (self.global_steps is not None) and (self.total_num_steps is not None):
            worker_id = torch.utils.data.get_worker_info().id
            self.tokenizer.attr_mask_ratio = (
                1 - (self.global_steps + worker_id) / self.total_num_steps
            )
            print(
                f"[worker_id {worker_id}] attr_mask_ratio: {self.tokenizer.attr_mask_ratio}, global_steps: {self.global_steps + worker_id}"
            ) if (self.global_steps + worker_id) % 100 == 0 else None
            self.global_steps += self.num_workers

        features = (
            [_add_idx_to_dict(self.tokenizer(graph), idx) for idx, graph in graphs]
            if isinstance(graphs[0], Tuple)
            else [_add_idx_to_dict(self.tokenizer(graph), 0) for graph in graphs]
        )
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
            mask_boundary=self.mask_boundary,
        )
        # print(f"[worker_id {worker_id}] attr_mask_ratio: {self.tokenizer.attr_mask_ratio}, global_steps: {self.global_steps+worker_id}, features ele shape: {features['input_ids'].shape}")
        # TODO: implement decoding input_ids
        return features


@dataclass
class DataCollatorForTokenizationPreprocessing:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: GSTTokenizer
    return_tensors: str = "pt"

    def __call__(self, graphs, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        features = [self.tokenizer(graph) for graph in graphs]
        return features
