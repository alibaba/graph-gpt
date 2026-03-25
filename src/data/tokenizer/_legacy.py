"""Backward-compatibility shim.

All tokenizer classes and helpers have moved to dedicated submodules
inside the ``src.data.tokenizer`` package.  This file re-exports every
public name so that code which imported from the old monolithic module
continues to work without changes.
"""

# Core classes
from .core import GSTTokenizer, StackedGSTTokenizer  # noqa: F401

# DICT_pos_func is now a class attribute of StackedGSTTokenizer
DICT_pos_func = StackedGSTTokenizer.DICT_pos_func

# Submodule re-exports (used by __init__.__getattr__)
from .graph_encoding import (  # noqa: F401
    _tokenize_discrete_attr,
    _tokenize_continuous_attr,
    _remove_lead_zero,
    _get_node2attr_mapping,
    _get_edge2attr_mapping,
    _get_graph2attr_mapping,
    get_semantics_attr_mapping,
    get_semantics_raw_node_edge2attr_mapping,
    mask_semantics_attr,
    mask_semantics_raw_node_edge_attr,
)
from .stacking import (  # noqa: F401
    add_eos_embed,
    stack_node_edge_graph_attr_to_node,
    stack_attr_to_node_and_edge,
    get_default_semantics_attr_mapping,
    get_default_semantics_embed_mapping,
)
from .padding import (  # noqa: F401
    _merge_two_ls,
    _get_batch_seq_len,
    get_input_dict_from_seq_tokens_id,
)
