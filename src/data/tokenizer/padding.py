"""Padding and batch construction utilities."""

import math
from typing import List, Set, Union


def _merge_two_ls(ls_main, ls_side, side="left"):
    return ls_side + ls_main if side == "left" else ls_main + ls_side


def _get_batch_seq_len(ls_seq_len, pad_to_multiple_of, max_length):
    if pad_to_multiple_of is None:
        batch_seq_len = max_length
    elif (
        len(ls_seq_len) == 1
    ):  # single sequence -> packed multiple samples => NO padding
        batch_seq_len = max(ls_seq_len)
    else:
        max_seq_len = max(ls_seq_len)
        batch_seq_len = pad_to_multiple_of * int(
            math.ceil(max_seq_len / pad_to_multiple_of)
        )
        batch_seq_len = min(batch_seq_len, max_length)
    return batch_seq_len


def get_input_dict_from_seq_tokens_id(
    seq_tokens_id: List[Union[int, List[int]]],
    seq_labels_id: List[Union[int, List[int]]],
    label_to_be_pad: Set[int],
    label_pad_token_id: int,
):
    seq_tokens_id = list(seq_tokens_id)
    if seq_labels_id is None:
        input_ids = seq_tokens_id[:-1]
        labels = seq_tokens_id[1:]
    else:
        seq_labels_id = list(seq_labels_id)
        input_ids = seq_tokens_id
        labels = seq_labels_id
    assert len(input_ids) == len(
        labels
    ), f"input_ids: {len(input_ids)}, labels: {len(labels)}"
    if len(label_to_be_pad) > 0:
        labels = [
            token_id if token_id not in label_to_be_pad else label_pad_token_id
            for token_id in labels
        ]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
