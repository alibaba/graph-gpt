"""Data types and constants for the tokenizer package."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from transformers.utils import ModelOutput


MOL_ENERGY_BIN_LEN = 16
MOL_ENERGY_SCALE = 1000


@dataclass
class TokenizationOutput(ModelOutput):
    """
    Base class for tokenizer's outputs.

    Args:
        ls_tokens: List of input tokens.
        ls_labels: List of label tokens.
        ls_raw_node_idx: List of raw node's index.
        tgt_node_token: Target node token for node-level tasks.
        tgt_edge_src_token: Target src node token for edge-level tasks.
        tgt_edge_dst_token: Target dst node token for edge-level tasks.
        tgt_edge_attr_token: Target edge attr token for edge-level tasks.
        tgt_pos: For UniBi attention mixed model.
        ls_embed: Embedding vectors.
        ls_len: Lengths for packed sequences.
    """

    ls_tokens: List[Union[str, List[str]]] = None
    ls_labels: List[Union[str, List[str]]] = None
    ls_raw_node_idx: List[int] = None
    tgt_node_token: Union[str, List[str], Tuple[str]] = None
    tgt_edge_src_token: Union[str, List[str], Tuple[str]] = None
    tgt_edge_dst_token: Union[str, List[str], Tuple[str]] = None
    tgt_edge_attr_token: List[str] = None
    tgt_pos: Optional[torch.Tensor] = None
    ls_embed: List[List[float]] = None
    ls_len: List[int] = None
