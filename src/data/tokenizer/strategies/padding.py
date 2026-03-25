"""Padding strategies for tokenized sequences."""

from abc import ABC, abstractmethod
from typing import Dict, List, Union
import numpy as np
import torch


class PaddingStrategy(ABC):
    """Abstract base class for padding strategies."""

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        padding_side: str = "right",
    ):
        assert padding_side in {"left", "right"}
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.padding_side = padding_side

    @abstractmethod
    def pad_batch(
        self,
        features: List[Dict],
        *,
        max_length: int,
        pad_to_multiple_of: int = 8,
        return_tensors: str = "pt",
        padding: Union[bool, str] = True,
    ) -> Dict:
        """Pad a batch of features to the same length."""
        pass

    @abstractmethod
    def pad_single(self, feature: Dict, pad_to: int) -> Dict:
        """Pad a single feature to target length."""
        pass


class FlatPaddingStrategy(PaddingStrategy):
    """Padding strategy for 1D (flat) token sequences."""

    def pad_batch(
        self,
        features: List[Dict],
        *,
        max_length: int = 128,
        pad_to_multiple_of: int = 8,
        return_tensors: str = "pt",
        padding: Union[bool, str] = True,
    ) -> Dict:
        from ..padding import _get_batch_seq_len

        assert return_tensors in {"pt", "np"}
        func = {"pt": torch.tensor, "np": np.array}[return_tensors]

        ls_seq_len = [len(feat["input_ids"]) for feat in features]
        pad_to = _get_batch_seq_len(ls_seq_len, pad_to_multiple_of, max_length)

        features = [self.pad_single(feat, pad_to) for feat in features]

        batch_outputs = {}
        for feat in features:
            for key, value in feat.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        _list_only_keys = {"sample_lens", "split_lens", "attn_modes"}
        batch_outputs = {
            key: (
                val
                if key in _list_only_keys
                else func(val)
                if not isinstance(val[0], str)
                else np.array(val)
            )
            for key, val in batch_outputs.items()
        }
        return batch_outputs

    def pad_single(self, feature: Dict, pad_to: int) -> Dict:
        from ..padding import _merge_two_ls

        if pad_to > len(feature["input_ids"]):
            padding_len = pad_to - len(feature["input_ids"])

            input_pad_val = self.pad_token_id
            label_pad_val = self.label_pad_token_id

            padded_input_ids = [input_pad_val] * padding_len
            padded_labels = [label_pad_val] * padding_len
            padded_position_ids = [0] * padding_len
            padded_attention_mask = [0] * padding_len

            feature["input_ids"] = _merge_two_ls(
                feature["input_ids"], padded_input_ids, self.padding_side
            )
            if "position_ids" in feature:
                feature["position_ids"] = _merge_two_ls(
                    feature["position_ids"], padded_position_ids, self.padding_side
                )
            feature["labels"] = _merge_two_ls(
                feature["labels"], padded_labels, self.padding_side
            )
            feature["attention_mask"] = _merge_two_ls(
                feature["attention_mask"], padded_attention_mask, self.padding_side
            )
            if "raw_node_idx" in feature:
                padded_nodev2_labels = [self.label_pad_token_id] * padding_len
                feature["raw_node_idx"] = _merge_two_ls(
                    feature["raw_node_idx"], padded_nodev2_labels, self.padding_side
                )
            for name in {"embed", "noise"}:
                if name in feature:
                    dim = len(feature[name][0])
                    padded_vecs = np.zeros(
                        (padding_len, dim), dtype=np.float32
                    ).tolist()
                    feature[name] = _merge_two_ls(
                        feature[name], padded_vecs, self.padding_side
                    )
        else:
            keys_set = {
                "input_ids",
                "position_ids",
                "labels",
                "nodev2_labels",
                "raw_node_idx",
                "attention_mask",
                "embed",
                "noise",
            }
            for key, val in feature.items():
                if key in keys_set:
                    if isinstance(val, np.ndarray):
                        feature[key] = val[:pad_to, :pad_to].tolist()
                    else:
                        feature[key] = val[:pad_to]
        return feature


class StackedPaddingStrategy(PaddingStrategy):
    """Padding strategy for 2D (stacked) token sequences."""

    def pad_batch(
        self,
        features: List[Dict],
        *,
        max_length: int = 128,
        pad_to_multiple_of: int = 8,
        return_tensors: str = "pt",
        padding: Union[bool, str] = True,
    ) -> Dict:
        from ..padding import _get_batch_seq_len

        assert return_tensors in {"pt", "np"}
        func = {"pt": torch.tensor, "np": np.array}[return_tensors]

        ls_seq_len = [len(feat["input_ids"]) for feat in features]
        pad_to = _get_batch_seq_len(ls_seq_len, pad_to_multiple_of, max_length)

        features = [self.pad_single(feat, pad_to) for feat in features]

        batch_outputs = {}
        for feat in features:
            for key, value in feat.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        _list_only_keys = {"sample_lens", "split_lens", "attn_modes"}
        batch_outputs = {
            key: (
                val
                if key in _list_only_keys
                else func(val)
                if not isinstance(val[0], str)
                else np.array(val)
            )
            for key, val in batch_outputs.items()
        }
        return batch_outputs

    def pad_single(self, feature: Dict, pad_to: int) -> Dict:
        from ..padding import _merge_two_ls

        if pad_to > len(feature["input_ids"]):
            padding_len = pad_to - len(feature["input_ids"])
            input_components = len(feature["input_ids"][0])
            label_components = len(feature["labels"][0])

            input_pad_val = [self.pad_token_id] * input_components
            label_pad_val = [self.label_pad_token_id] * label_components

            padded_input_ids = [input_pad_val] * padding_len
            padded_labels = [label_pad_val] * padding_len
            padded_position_ids = [0] * padding_len
            padded_attention_mask = [0] * padding_len

            feature["input_ids"] = _merge_two_ls(
                feature["input_ids"], padded_input_ids, self.padding_side
            )
            if "position_ids" in feature:
                feature["position_ids"] = _merge_two_ls(
                    feature["position_ids"], padded_position_ids, self.padding_side
                )
            feature["labels"] = _merge_two_ls(
                feature["labels"], padded_labels, self.padding_side
            )
            feature["attention_mask"] = _merge_two_ls(
                feature["attention_mask"], padded_attention_mask, self.padding_side
            )
            if "raw_node_idx" in feature:
                padded_nodev2_labels = [self.label_pad_token_id] * padding_len
                feature["raw_node_idx"] = _merge_two_ls(
                    feature["raw_node_idx"], padded_nodev2_labels, self.padding_side
                )
            for name in {"embed", "noise"}:
                if name in feature:
                    dim = len(feature[name][0])
                    padded_vecs = np.zeros(
                        (padding_len, dim), dtype=np.float32
                    ).tolist()
                    feature[name] = _merge_two_ls(
                        feature[name], padded_vecs, self.padding_side
                    )
        else:
            keys_set = {
                "input_ids",
                "position_ids",
                "labels",
                "nodev2_labels",
                "raw_node_idx",
                "attention_mask",
                "embed",
                "noise",
            }
            for key, val in feature.items():
                if key in keys_set:
                    if isinstance(val, np.ndarray):
                        feature[key] = val[:pad_to, :pad_to].tolist()
                    else:
                        feature[key] = val[:pad_to]
        return feature
