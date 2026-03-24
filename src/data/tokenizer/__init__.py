"""
tokenizer package -- Graph Sequence Tokenization.

This __init__.py uses lazy loading to avoid circular imports.
Submodules like ``types.py`` can be imported directly without triggering
the heavy ``_legacy`` module. Attributes from ``_legacy`` are loaded on
first access via ``__getattr__``, which preserves existing import patterns:

- ``from ..data import tokenizer`` then ``getattr(tokenizer, "GSTTokenizer")``
- ``from .tokenizer import GSTTokenizer, StackedGSTTokenizer``

New strategy-based classes are available directly:
- ``from .tokenizer import BaseTokenizer``
- ``from .tokenizer.strategies import PaddingStrategy, TaskPreparationStrategy``
"""

# Core classes - available directly (no lazy loading needed for new architecture)
from .base import BaseTokenizer
from .core import GSTTokenizer, StackedGSTTokenizer

# Strategy classes - available directly
from .strategies import (
    PaddingStrategy,
    FlatPaddingStrategy,
    StackedPaddingStrategy,
    SequencePacker,
    TaskPreparationStrategy,
    get_task_strategy,
    PretrainMLMStrategy,
    PretrainCoordStrategy,
    GraphLevelStrategy,
    EdgeLevelStrategy,
    NodeLevelStrategy,
    NodeV2Strategy,
)

__all__ = [
    # Core classes
    "BaseTokenizer",
    "GSTTokenizer",
    "StackedGSTTokenizer",
    # Strategy classes
    "PaddingStrategy",
    "FlatPaddingStrategy",
    "StackedPaddingStrategy",
    "SequencePacker",
    "TaskPreparationStrategy",
    "get_task_strategy",
    "PretrainMLMStrategy",
    "PretrainCoordStrategy",
    "GraphLevelStrategy",
    "EdgeLevelStrategy",
    "NodeLevelStrategy",
    "NodeV2Strategy",
    # Legacy exports (lazy loaded)
    "DICT_pos_func",
    "get_semantics_raw_node_edge2attr_mapping",
    "get_semantics_attr_mapping",
    "mask_semantics_attr",
    "mask_semantics_raw_node_edge_attr",
    "_tokenize_discrete_attr",
    "_tokenize_continuous_attr",
    "_remove_lead_zero",
    "_add_regression_token",
    "_get_node2attr_mapping",
    "_get_edge2attr_mapping",
    "_get_graph2attr_mapping",
    "_merge_two_ls",
    "_get_batch_seq_len",
    "get_input_dict_from_seq_tokens_id",
    "stack_node_edge_graph_attr_to_node",
    "stack_attr_to_node_and_edge",
    "add_eos_embed",
    "get_default_semantics_attr_mapping",
    "get_default_semantics_embed_mapping",
]

# Lazy loading: _legacy is only imported when a legacy attribute is first accessed.
# This breaks the circular import chain:
#   tokenizer_utils -> tokenizer.types (standalone, no _legacy dependency)
#   tokenizer.__init__ -> does NOT eagerly load _legacy

_legacy_module = None
_legacy_names = {
    "DICT_pos_func",
    "get_semantics_raw_node_edge2attr_mapping",
    "get_semantics_attr_mapping",
    "mask_semantics_attr",
    "mask_semantics_raw_node_edge_attr",
    "_tokenize_discrete_attr",
    "_tokenize_continuous_attr",
    "_remove_lead_zero",
    "_add_regression_token",
    "_get_node2attr_mapping",
    "_get_edge2attr_mapping",
    "_get_graph2attr_mapping",
    "_merge_two_ls",
    "_get_batch_seq_len",
    "get_input_dict_from_seq_tokens_id",
    "stack_node_edge_graph_attr_to_node",
    "stack_attr_to_node_and_edge",
    "add_eos_embed",
    "get_default_semantics_attr_mapping",
    "get_default_semantics_embed_mapping",
}


def _load_legacy():
    global _legacy_module
    if _legacy_module is None:
        from . import _legacy

        _legacy_module = _legacy
    return _legacy_module


def __getattr__(name):
    # Direct exports are already loaded above
    if name in ("BaseTokenizer", "GSTTokenizer", "StackedGSTTokenizer"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _legacy_names:
        return getattr(_load_legacy(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
