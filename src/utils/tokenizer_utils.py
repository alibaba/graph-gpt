"""Backward-compatibility shim.

All tokenizer utility code has moved to ``src.data.tokenizer.*`` submodules.
This file re-exports every public name so that existing import paths
(``from src.utils.tokenizer_utils import ...``) continue to work.
"""

# Re-export types and constants
from src.data.tokenizer.types import (  # noqa: F401
    TokenizationOutput,
    MOL_ENERGY_BIN_LEN,
    MOL_ENERGY_SCALE,
)

# Re-export masking utilities
from src.data.tokenizer.masking import (  # noqa: F401
    _mask_ids,
    _get_keys,
    _mask_stacked_input_ids,
    _mask_stacked_input_ids_v2,
    _mask_stacked_input_ids_dlm,
    _mask_input_ids,
    _pad_stacked_targets,
    get_mask_of_raw_seq,
)
