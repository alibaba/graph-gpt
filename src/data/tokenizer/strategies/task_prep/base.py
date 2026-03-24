"""Base class for task preparation strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Union, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import Data
    from ...types import TokenizationOutput


class TaskPreparationStrategy(ABC):
    """Abstract base for task-specific input preparation."""

    @abstractmethod
    def prepare(
        self,
        in_dict: Dict,
        token_res: "TokenizationOutput",
        graph: "Data",
        gtokenizer,
    ) -> Dict:
        """
        Prepare inputs for specific task type.

        Args:
            in_dict: Dictionary with input_ids, labels, attention_mask
            token_res: Tokenization output with metadata
            graph: Original PyG Data object
            gtokenizer: Reference to tokenizer for vocab access

        Returns:
            Modified in_dict with task-specific fields
        """
        pass

    def _map_tokens_to_ids(
        self, tokens: Union[str, Iterable[str], None], vocab_map: Dict
    ):
        """Helper to map tokens to IDs."""
        if tokens is None:
            return None
        elif isinstance(tokens, str):
            return vocab_map[tokens]
        elif isinstance(tokens, (list, tuple)):
            return [vocab_map[t] for t in tokens]
        else:
            raise ValueError(f"Unsupported token type: {type(tokens)}")

    def _extend_input_dict(
        self,
        in_dict: Dict,
        ls_extend_tokens,
        keys=tuple(),
        vals=tuple(),
    ) -> Dict:
        """Extend input dict with additional tokens."""
        len_extended_tokens = len(ls_extend_tokens)
        inputs_instance = in_dict["input_ids"][0]

        # Handle 2D (stacked) tokens
        if isinstance(inputs_instance, list):
            token_dim = len(inputs_instance)
            ls_extend_tokens = [
                [token_id] * token_dim if isinstance(token_id, int) else token_id
                for token_id in ls_extend_tokens
            ]

        in_dict["input_ids"].extend(ls_extend_tokens)

        labels_instance = in_dict["labels"][0]
        if isinstance(labels_instance, list):
            ls_extend_labels = [[-100] * len(labels_instance)] * len_extended_tokens
        else:
            ls_extend_labels = [-100] * len_extended_tokens

        in_dict["labels"].extend(ls_extend_labels)
        in_dict["attention_mask"].extend([1] * len_extended_tokens)

        for key, val in zip(keys, vals):
            in_dict[key].extend([val] * len_extended_tokens)

        return in_dict
