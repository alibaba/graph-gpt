"""Masking strategies for token-level prediction tasks."""

import random
import math
from typing import List, Tuple, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Core masking helpers
# ---------------------------------------------------------------------------


def _mask_ids(
    mask_token_id,
    global_rnd_id,
    raw_id,
    mask_token_precent: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    pad_token_id: int = 0,
):
    # If the i-th token is chosen, we replace the i-th token with
    # (1) the [MASK] token 80% of the time
    # (2) a random token 10% of the time
    # (3) the unchanged i-th token 10% of the time
    rate_vec = np.cumsum(mask_token_precent)
    assert rate_vec[2] == 1.0, f"rate_vec: {rate_vec}"
    if raw_id == pad_token_id:
        return raw_id
    rnd = random.random()
    if rnd < rate_vec[0]:
        return mask_token_id
    elif rnd < rate_vec[1]:
        return global_rnd_id
    else:
        return raw_id


def _get_keys(idx, ls: List[int], ls_of_ls: List[List[int]]):
    if idx % 2 == 0:  # key for node
        key = ls[0]
    else:  # key for edge
        prev_node = ls_of_ls[idx - 1][0]
        next_node = ls_of_ls[idx + 1][0]
        key = (
            (prev_node, next_node) if prev_node < next_node else (next_node, prev_node)
        )
    return key


# ---------------------------------------------------------------------------
# Masking strategies
# ---------------------------------------------------------------------------
def _get_mask_ratio_batch(
    conf: dict,
    gtokenizer,
    num_sequences: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Vectorized mask ratio generation for multiple sequences.
    """
    if conf["name"] == "fixed":
        alpha_t = conf["params"]["fixed_ratio"]
        mask_ratios = np.full(num_sequences, alpha_t)
        wgts = None

    elif conf["name"] == "polynomial":
        powers = conf["params"]["power"]
        umr_min, umr_max = gtokenizer.train_cfg.pretrain_mlm.params.umr_clip

        # Vectorized random generation
        r = np.random.uniform(umr_min, umr_max, size=num_sequences)
        t = r
        mask_ratios = 1 - t**powers

        # Compute weights
        alpha_t_prime = -powers * t ** (powers - 1)  # type: ignore
        wgts = powers / t

    else:  # cosine
        r = np.random.rand(num_sequences)
        mask_ratios = np.cos(r * np.pi) * 0.5 + 0.5
        wgts = None

    return mask_ratios, wgts


def _mask_input_ids_unified(
    input_ids: Union[List[int], List[List[int]]],
    mask_token_id: int,
    all_vocab_ids: List[int],
    mask_ratio: Union[float, np.ndarray],  # scalar or per-element ratio
    mask_token_precent: Tuple[float, float, float] = (1, 0, 0),
    pad_token_id: int = 0,
    **kwargs,
) -> Tuple[Union[List[int], List[List[int]]], Union[List[int], List[List[int]]]]:
    """
    Unified vectorized masking for both 1D and 2D inputs.

    Args:
        input_ids: Flat list [seq] or stacked list [[seq, dim], ...]
        mask_ratio: Scalar (float) or array with same shape as input_ids
                   for per-position masking ratios
        ...

    Returns:
        (masked_input_ids, labels_mask) with same structure as input
    """
    # Convert to numpy array (handles both 1D and 2D)
    arr = np.array(input_ids)
    original_shape = arr.shape

    # Ensure mask_ratio is broadcastable to input shape
    if np.isscalar(mask_ratio):
        mask_ratio_arr = np.full(original_shape, mask_ratio)
    else:
        mask_ratio_arr = np.array(mask_ratio)
        assert (
            mask_ratio_arr.shape[0] == original_shape[0]
        ), f"mask_ratio shape {mask_ratio_arr.shape[0]} != input shape {original_shape[0]}"

    # Initialize labels mask
    labels_mask = np.full(original_shape, -100)

    # Vectorized mask selection with per-element ratios
    rand = np.random.rand(*original_shape)
    mask = rand < mask_ratio_arr

    # Save original values to labels
    labels_mask = np.where(mask, arr, labels_mask)

    # Vectorized masking strategy
    rate_vec = np.cumsum(mask_token_precent)
    rnd = np.random.rand(*original_shape)

    # Strategy 1: Replace with [MASK]
    replace_mask = mask & (rnd < rate_vec[0]) & (arr != pad_token_id)
    arr = np.where(replace_mask, mask_token_id, arr)

    # Strategy 2: Replace with random token
    if rate_vec[1] > rate_vec[0]:
        random_mask = (
            mask & (rnd >= rate_vec[0]) & (rnd < rate_vec[1]) & (arr != pad_token_id)
        )
        random_tokens = np.random.choice(all_vocab_ids, size=original_shape)
        arr = np.where(random_mask, random_tokens, arr)

    # Convert back to original format
    return arr.tolist(), labels_mask.tolist()


def _get_mask_ratio(conf, gtokenizer):
    wgt = None
    if conf["name"] == "fixed":
        alpha_t = conf["params"]["fixed_ratio"]
    elif conf["name"] == "polynomial":
        # 3-> cubic, 2-> square, 1-> linear, 0.5-> sqrt
        powers = conf["params"]["power"]
        umr_min, umr_max = gtokenizer.train_cfg.pretrain_mlm.params.umr_clip
        assert 0 <= umr_min <= umr_max <= 1
        r = random.random()
        t = umr_min + (umr_max - umr_min) * r  # rescale to [mr_min, mr_max]
        alpha_t = 1 - t**powers
        alpha_t_prime = -powers * t ** (powers - 1)  # type: ignore
        # Fig. 1 @ https://arxiv.org/pdf/2406.04329
        wgt = powers / t  # - alpha_t_prime / (1 - alpha_t)
    else:
        alpha_t = math.cos(random.random() * math.pi) * 0.5 + 0.5
    return alpha_t, wgt


def _mask_stacked_input_ids(
    input_ids: List[List[int]],
    mask_token_id,
    all_vocab_ids,
    mask_ratio: float = 0.15,
    mask_token_precent: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    pad_token_id: int = 0,
    has_eos: bool = True,
    stack_method: str = "short",
):
    labels_mask = np.full((len(input_ids), len(input_ids[0])), -100).tolist()
    if has_eos:
        eos = input_ids[-1:]
        input_ids = input_ids[:-1]
    else:
        eos = []
        input_ids = input_ids
    if stack_method == "short":
        keys = [ele[0] for ele in input_ids]
    else:
        assert (
            len(input_ids) % 2 == 1
        ), f"tmp_ids: {input_ids},\nhas_eos: {has_eos},\n{locals()}"
        keys = [_get_keys(i, ele, input_ids) for i, ele in enumerate(input_ids)]
    keys_set = set(keys)
    keys_masked = random.sample(
        list(keys_set), k=int(np.ceil(len(keys_set) * mask_ratio))
    )
    keys_masked_set = set(keys_masked)

    for idx, (ls_tokens, key) in enumerate(zip(input_ids, keys)):
        if key in keys_masked_set:
            labels_mask[idx] = input_ids[idx]
            input_ids[idx] = [
                _mask_ids(
                    mask_token_id,
                    random.sample(all_vocab_ids, k=1)[0],
                    ele,
                    mask_token_precent,
                    pad_token_id,
                )
                for ele in ls_tokens
            ]
    input_ids.extend(eos)
    return input_ids, labels_mask


def _mask_stacked_input_ids_v2(
    input_ids: List[List[int]],
    mask_token_id,
    all_vocab_ids,
    mask_ratio: float = 0.15,
    mask_token_precent: Tuple[float, float, float] = (1, 0, 0),
    pad_token_id: int = 0,
    **kwargs,
):
    # v2 choose tokens to mask globally
    seq = len(input_ids)
    dim = len(input_ids[0])
    input_ids = np.array(input_ids)
    labels_mask = np.full((seq, dim), -100)

    indices = list(np.ndindex((seq, dim)))
    idx_masked = random.sample(
        range(len(indices)), k=int(np.ceil(len(indices) * mask_ratio))
    )
    rate_vec = np.cumsum(mask_token_precent)

    for idx in idx_masked:
        idx_seq, idx_dim = indices[idx]
        labels_mask[idx_seq, idx_dim] = input_ids[idx_seq, idx_dim]
        if input_ids[idx_seq, idx_dim] != pad_token_id:
            rnd = random.random()
            if rnd < rate_vec[0]:
                input_ids[idx_seq, idx_dim] = mask_token_id
            elif rnd < rate_vec[1]:
                input_ids[idx_seq, idx_dim] = random.sample(all_vocab_ids, k=1)[0]
            else:
                pass
    return input_ids.tolist(), labels_mask.tolist()


def _mask_stacked_input_ids_dlm(
    input_ids: List[List[int]],
    mask_token_id,
    all_vocab_ids,
    mask_ratio: float = 0.15,
    mask_token_precent: Tuple[float, float, float] = (1, 0, 0),
    pad_token_id: int = 0,
    has_eos: bool = True,
    stack_method: str = "short",
):
    # dLM-type of masking
    seq = len(input_ids)
    dim = len(input_ids[0])
    input_ids = np.array(input_ids)
    labels_mask = np.full((seq, dim), -100)
    rand = np.random.rand(seq, dim)  # [0,1)
    mask = rand < mask_ratio
    mask = mask & (input_ids != pad_token_id)

    labels_mask = np.where(mask, input_ids, labels_mask)
    input_ids = np.where(mask, mask_token_id, input_ids)
    return input_ids.tolist(), labels_mask.tolist()


def _mask_input_ids(
    input_ids: List[int],
    mask_token_id,
    all_vocab_ids,
    mask_ratio: float = 0.15,
    mask_token_precent: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    pad_token_id: int = 0,
):
    labels_mask = [-100] * len(input_ids)
    idx_masked = random.sample(
        range(len(input_ids)), k=int(np.ceil(len(input_ids) * mask_ratio))
    )
    rate_vec = np.cumsum(mask_token_precent)
    assert rate_vec[2] == 1.0, f"rate_vec: {rate_vec}"
    for idx in idx_masked:
        labels_mask[idx] = input_ids[idx]
        if input_ids[idx] != pad_token_id:
            rnd = random.random()
            if rnd < rate_vec[0]:
                input_ids[idx] = mask_token_id
            elif rnd < rate_vec[1]:
                input_ids[idx] = random.sample(all_vocab_ids, k=1)[0]
            else:
                pass
    return input_ids, labels_mask


def _pad_stacked_targets(
    i, ls_token_ids, *, node_attr_dim=9, padding_val=-100, eos_token_id=None
):
    if i % 2 == 0:  # pad node labels
        ls_token_ids = [
            token_id if j <= node_attr_dim else padding_val
            for j, token_id in enumerate(ls_token_ids)
        ]
    else:  # pad edge labels
        ls_token_ids = [
            token_id if (j > node_attr_dim or token_id == eos_token_id) else padding_val
            for j, token_id in enumerate(ls_token_ids)
        ]
    return ls_token_ids


# ---------------------------------------------------------------------------
# Raw-sequence mask generation (Eulerian path deduplication)
# ---------------------------------------------------------------------------


def _obtain_all_idx_of_each_element(seq: List):
    dict_idx = {}
    for i, ele in enumerate(seq):
        if ele not in dict_idx:
            dict_idx[ele] = []
        dict_idx[ele].append(i)
    return dict_idx


def _obtain_first_appearance_idx(dict_idx):
    return [val[0] for val in dict_idx.values()]


def _obtain_last_appearance_idx(dict_idx):
    return [val[-1] for val in dict_idx.values()]


def _obtain_random_appearance_idx(dict_idx):
    return [random.choice(val) for val in dict_idx.values()]


def _obtain_all_appearance_idx(dict_idx):
    return [idx for val in dict_idx.values() for idx in val]


DICT_MASK_FUNC = {
    "first": _obtain_first_appearance_idx,
    "last": _obtain_last_appearance_idx,
    "random": _obtain_random_appearance_idx,
    "all": _obtain_all_appearance_idx,
}


def get_mask_of_raw_seq(raw_seq, mask_type="first"):
    deco_seq = [
        (min(ele), max(ele)) if isinstance(ele, tuple) else ele for ele in raw_seq
    ]
    dict_deco_idx = _obtain_all_idx_of_each_element(deco_seq)
    mask_type = (
        random.choice(("first", "last", "random")) if mask_type == "mix" else mask_type
    )
    mask_func = DICT_MASK_FUNC[mask_type]
    idx = mask_func(dict_deco_idx)
    idx = sorted(idx)

    seq_len = len(raw_seq)
    mask = np.zeros(seq_len, dtype=int)
    mask[idx] = 1
    return mask
