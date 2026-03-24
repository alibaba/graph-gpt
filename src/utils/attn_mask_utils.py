from typing import List, Union, Tuple, Optional
import torch
from packaging import version

parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)
is_torch_greater_or_equal_than_1_13 = parsed_torch_version_base >= version.parse("1.13")


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


# check https://aliyuque.antfin.com/james.zqf/ssqcu1/dexa1q0g8givelio?singleDoc# for `_prepare_4d_bi_causal_attention_mask` implementation details

# refer to: `transformers/modeling_attn_mask_utils.py::_prepare_4d_causal_attention_mask`
# @ transformers==4.36.2
# check https://aliyuque.antfin.com/james.zqf/ssqcu1/dexa1q0g8givelio?singleDoc# for `_prepare_4d_causal_bi_attention_mask` implementation details
