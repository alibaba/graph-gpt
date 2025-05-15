from typing import Optional, Union, Tuple, List
import torch
from packaging import version

parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)
is_torch_greater_or_equal_than_1_13 = parsed_torch_version_base >= version.parse("1.13")


# refer to: `transformers/modeling_attn_mask_utils.py::_prepare_4d_causal_attention_mask`
# @ transformers==4.36.2
# check https://aliyuque.antfin.com/james.zqf/ssqcu1/dexa1q0g8givelio?singleDoc# for implementation details
def _prepare_4d_causal_bi_attention_mask(
    attention_mask: Optional[torch.Tensor],
    attention_mask_bi: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    boundary_mask_idx: Optional[torch.Tensor] = None,
):
    """
    Creates a causal & bi mixed 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask for causal of shape `(batch_size, key_value_length)`
        attention_mask_bi (`torch.Tensor` or `None`):
            A 2D attention mask for bidirectional of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
    """
    assert (
        past_key_values_length == 0
    ), "NOT implemented, refer to transformers[4.36.2]/modeling_attn_mask_utils.py::_prepare_4d_causal_attention_mask"
    if attention_mask_bi is None:
        attention_mask_bi = torch.zeros_like(attention_mask)

    dtype = inputs_embeds.dtype
    device = attention_mask.device
    bsz, tgt_len = input_shape

    # 4d mask is passed through the layers
    # A). prepare for causal & bi-directional attention
    # 1. obtain basic mask matrix
    mask = torch.full((bsz, tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    # [bsz, len, len] of min values, e.g., -3.4028e+38 for float32

    # 2. obtain the causal mask matrix -> lower triangle of T/F
    mask_cond = torch.arange(mask.size(-1), device=device)  # [len]
    tri_tf = mask_cond < (mask_cond + 1).view(mask.size(-1), 1)
    # [len] & [len, 1] -> [len, len], lower triangle of True-False

    # 3. obtain the bi-directional mask matrix
    expanded_mask = (
        attention_mask_bi[:, None, :].expand(bsz, tgt_len, tgt_len).to(torch.bool)
    )  # [bsz, len] -> [bsz, len, len]
    expanded_mask = expanded_mask.transpose(1, 2) & expanded_mask

    # 2&3. merge causal & bi mask
    uni_bi_mask = expanded_mask | tri_tf[None, :, :]  # [bsz, len, len] of True/False

    # 2&3. mask boundary
    if boundary_mask_idx is not None:
        s_idx = boundary_mask_idx
        uni_bi_mask[s_idx[:, 0], s_idx[:, 1], s_idx[:, 2]] = False
        # TODO: find more elegant way to mask boundary

    # 4. un-mask some entries in mask
    mask = mask.masked_fill_(uni_bi_mask, 0)
    mask = mask.to(dtype)
    mask = mask.unsqueeze(1)  # [bsz, len, len] -> [bsz, 1, len, len]

    # B). prepare for padding mask as in attention_mask
    inv_attn_mask = (1 - attention_mask).bool()  # [bsz, len]
    expanded_mask = inv_attn_mask[:, None, None, :].expand(bsz, 1, tgt_len, tgt_len)
    # [bsz, len] -> [bsz, 1, len, len]

    # C). merge to get final 4-D mask
    mask = mask.masked_fill(expanded_mask, torch.finfo(dtype).min)
    return mask


def get_masked_boundary_idx(
    attention_mask: torch.LongTensor, attention_mask_bi: torch.LongTensor, tgt_len: int
):
    idx_boundary = attention_mask.sum(-1) - attention_mask_bi.sum(-1) - 1
    # [bsz]
    ls = []
    for i, idx in enumerate(idx_boundary.tolist()):
        ls_tmp = [(i, ele, idx) for ele in range(idx + 1, tgt_len)]
        ls.extend(ls_tmp)
    s_idx = torch.tensor(ls, dtype=torch.int64)
    return s_idx


# refer to: `transformers/modeling_attn_mask_utils.py::_prepare_4d_attention_mask`
# @ transformers==4.36.2
def _prepare_4d_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    dtype = inputs_embeds.dtype
    bsz, tgt_len = input_shape
    src_len = tgt_len
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = (
        attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    )

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


# check https://aliyuque.antfin.com/james.zqf/ssqcu1/dexa1q0g8givelio?singleDoc# for implementation details
def _prepare_4d_bi_causal_attention_mask(
    causal_len: Union[int, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
    tgt_len: Optional[int] = None,
):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # causal_len as 1st arg for using `partial` to create a new func when it's int
    device = attention_mask.device
    bsz, tgt_len = attention_mask.shape
    src_len = tgt_len

    mask = torch.arange(tgt_len, device=device)[None, :]
    # 1. create bi-attn mask
    bi_mask = mask < attention_mask.sum(dim=-1, keepdim=True) - causal_len
    bi_mask_4d = bi_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
    # 2. create causal mask
    causal_mask = mask < mask.view((-1, 1)) + 1
    causal_mask_4d = causal_mask[None, None, :, :].expand(bsz, 1, tgt_len, src_len)
    # 3. merge bi-/causal- masks
    bi_causal_mask_4d = (
        bi_mask_4d | causal_mask_4d & attention_mask.to(bool)[:, None, None, :]
    )
    # 4. to dtype
    inverted_mask = 1.0 - bi_causal_mask_4d.to(dtype)
    return inverted_mask.masked_fill(~bi_causal_mask_4d, torch.finfo(dtype).min)
