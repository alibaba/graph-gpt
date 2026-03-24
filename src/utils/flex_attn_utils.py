"""
Flex attention utilities for GraphGPT.

Functions adapted from holon/data/data_utils.py to support the split_lens/attn_modes
attention mask abstraction. These utilities can produce either:
  - SDPA path: per-sample 2D attention masks (prepare_attention_mask_per_sample)
  - Flex path: mask_mod closures for torch.nn.attention.flex_attention (create_sparse_mask)
"""

from typing import List, Optional

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
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (
            full_and_noise_seq_id[q_idx] >= 0
        )

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return ~(
            (noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])
        )

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ["full", "noise"] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == "noise" else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.tensor(full_and_noise_tmp, dtype=torch.int32).to(
        device
    )
    noise_seq_id = torch.tensor(noise_tmp, dtype=torch.int32).to(device)

    document_id = torch.cat(
        [
            torch.full((l,), i, device=device, dtype=torch.int32)
            for i, l in enumerate(document_lens, start=1)
        ]
    )

    return and_masks(
        or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask
    )


# ---------------------------------------------------------------------------
# Dispatcher helpers: build flex attn masks from split_lens/attn_modes
# ---------------------------------------------------------------------------
def build_flex_block_mask(
    num_heads: Optional[int],
    sample_lens: List[List[int]],
    split_lens: List[List[int]],
    attn_modes: List[List[str]],
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
):
    """Build a BlockMask from split_lens/attn_modes (flex attention path).

    Args:
        sample_lens: list of list of int — per-sample document/graph lengths
        split_lens: list of list of int — per-sample split lengths
        attn_modes: list of list of str — per-sample attention modes
        attention_mask: [bsz, seq_len] 1D padding mask
        input_tensor: [bsz, seq_len, dim]

    Returns:
        BlockMask for flex_attention, or falls back to 4D tensor if not on CUDA.
    """
    device = input_tensor.device

    from torch.nn.attention.flex_attention import create_block_mask

    # Compute document_lens (per-sample total lengths) and flatten splits
    assert len(sample_lens) == 1, f"bsz == {len(sample_lens)} != 1"
    document_lens = sample_lens[0]
    flat_split_lens = split_lens[0]
    flat_attn_modes = attn_modes[0]

    mask_mod = create_sparse_mask(
        document_lens, flat_split_lens, flat_attn_modes, device
    )
    seq_len = sum(document_lens)
    block_mask = create_block_mask(
        mask_mod,
        B=1,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        BLOCK_SIZE=128,
        _compile=True,
    )
    return block_mask
