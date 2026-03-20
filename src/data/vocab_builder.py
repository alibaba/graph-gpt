"""Backward-compatibility shim.

All vocab building code has moved to ``src.data.tokenizer.vocab``.
This file re-exports every public name so that existing import paths
(``from src.data.vocab_builder import ...``) continue to work.
"""

from .tokenizer.vocab import (  # noqa: F401
    load_vocab,
    build_vocab,
    save_vocab,
    get_structure_vocab,
    get_semantics_vocab,
)
