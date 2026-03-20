"""
Flex attention utilities for GraphGPT.

Functions adapted from holon/data/data_utils.py to support the split_lens/attn_modes
attention mask abstraction. These utilities can produce either:
  - SDPA path: per-sample 2D attention masks (prepare_attention_mask_per_sample)
  - Flex path: mask_mod closures for torch.nn.attention.flex_attention (create_sparse_mask)
"""

from typing import List

import torch
from torch.nn.attention.flex_attention import and_masks, or_masks


# ---------------------------------------------------------------------------
# Flex attention mask utilities (from holon)
# ---------------------------------------------------------------------------

def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    """Create ID tensors and a flat mask_mod closure for flex_attention.

    Combines ID creation and mask closure into one function, following the
    holon reference implementation.

    Args:
        document_lens: list[int] — length of each document/sample in packed sequence
        split_lens: list[int] — length of each split (flat across all documents)
        attn_modes: list[str] — attention mode per split ('causal', 'full', 'noise')
        device: torch.device

    Returns:
        A mask_mod function with signature (b, h, q_idx, kv_idx) -> bool
        that encodes: (causal OR same_full_split) AND same_document.
    """
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return (~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ['full', 'noise'] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == 'noise' else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat([torch.full((l,), i, device=device) for i, l in enumerate(document_lens, start=1)])

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """Create a 2D attention mask for a single sample (SDPA path).

    Args:
        split_lens: list[int] — length of each split within one sample
        attn_modes: list[str] — attention mode per split ('causal', 'full', 'noise')
        device: torch.device

    Returns:
        2D float tensor of shape (sample_len, sample_len) where
        0 = attend, -inf = mask.
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ['causal', 'full', 'noise']
        if attn_mode == "causal":
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones(
                (s, s), device=device
            ).tril()
            # Causal splits can attend to ALL previous positions (bi_causal prefix)
            attention_mask[csum:csum + s, :csum] = 1
        else:
            # Full/noise splits only attend within themselves (block-diagonal)
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones(
                (s, s), device=device
            )
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum:csum + s] = torch.zeros(
                (sample_len, s), device=device
            )
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones(
                (s, s), device=device
            )
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )

    return attention_mask


# ---------------------------------------------------------------------------
# Dispatcher helpers: build masks from split_lens/attn_modes
# ---------------------------------------------------------------------------

def build_4d_from_splits(
    split_lens: List[List[int]],
    attn_modes: List[List[str]],
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """Build a 4D attention mask from split_lens/attn_modes (SDPA path).

    Args:
        split_lens: list of list of int — per-sample split lengths [bsz][n_splits]
        attn_modes: list of list of str — per-sample attention modes [bsz][n_splits]
        attention_mask: [bsz, seq_len] 1D padding mask (1=valid, 0=pad)
        input_tensor: [bsz, seq_len, dim] for dtype reference

    Returns:
        4D tensor [bsz, 1, seq_len, seq_len] with 0=attend, min_float=mask.
    """
    dtype = input_tensor.dtype
    device = input_tensor.device
    bsz, seq_len = attention_mask.shape

    masks = []
    for b in range(bsz):
        mask_2d = prepare_attention_mask_per_sample(
            split_lens[b], attn_modes[b], device=device
        )
        # mask_2d is (sum(split_lens[b]), sum(split_lens[b]))
        # May be smaller than seq_len if split_lens doesn't include padding
        m_len = mask_2d.shape[0]
        if m_len < seq_len:
            # Extend with -inf for padding region
            full_mask = torch.full(
                (seq_len, seq_len), float("-inf"), dtype=torch.float, device=device
            )
            full_mask[:m_len, :m_len] = mask_2d
            mask_2d = full_mask
        masks.append(mask_2d)

    # Stack: [bsz, seq, seq] -> [bsz, 1, seq, seq]
    mask_4d = torch.stack(masks, dim=0).unsqueeze(1)
    return mask_4d.to(dtype)


def build_flex_block_mask(
    split_lens: List[List[int]],
    attn_modes: List[List[str]],
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
):
    """Build a BlockMask from split_lens/attn_modes (flex attention path).

    Args:
        split_lens: list of list of int — per-sample split lengths
        attn_modes: list of list of str — per-sample attention modes
        attention_mask: [bsz, seq_len] 1D padding mask
        input_tensor: [bsz, seq_len, dim]

    Returns:
        BlockMask for flex_attention, or falls back to 4D tensor if not on CUDA.
    """
    device = input_tensor.device
    if not torch.cuda.is_available() or device.type != 'cuda':
        return build_4d_from_splits(split_lens, attn_modes, attention_mask, input_tensor)

    from torch.nn.attention.flex_attention import create_block_mask

    bsz = attention_mask.shape[0]
    seq_len = attention_mask.shape[1]

    # Compute document_lens (per-sample total lengths) and flatten splits
    document_lens = [sum(sl) for sl in split_lens]
    flat_split_lens = []
    flat_attn_modes = []
    for sl, am in zip(split_lens, attn_modes):
        flat_split_lens.extend(sl)
        flat_attn_modes.extend(am)

    mask_mod = create_sparse_mask(document_lens, flat_split_lens, flat_attn_modes, device)

    block_mask = create_block_mask(
        mask_mod,
        B=bsz,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        BLOCK_SIZE=128,
        _compile=True,
    )
    return block_mask
