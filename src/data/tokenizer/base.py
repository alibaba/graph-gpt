"""Base tokenizer class with composition-based architecture."""

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional
from torch_geometric.data import Data

from .types import TokenizationOutput
from ..vocab_builder import load_vocab
from ...conf import TASK_TYPES


class BaseTokenizer(ABC):
    """
    Abstract base class for graph tokenizers.

    Uses composition for:
    - PaddingStrategy: handles sequence padding
    - SequencePacker: handles sequence packing (optional)
    - TaskPreparationStrategy: handles task-specific preparation
    """

    def __init__(
        self,
        config: Dict,
        *,
        padding_strategy=None,
        sequence_packer=None,
        task_preparer=None,
        add_eos: bool = True,
        train_cfg=None,
        **kwargs,
    ):
        self.config = config
        self.padding_strategy = padding_strategy
        self.sequence_packer = sequence_packer
        self.task_preparer = task_preparer
        self.add_eos = add_eos
        self.train_cfg = train_cfg
        self.kwargs = kwargs

        # Vocabulary management
        self.vocab_map = self._load_vocab()
        self.vocab_size = max(self.vocab_map.values()) + 1
        self.pad_token_id = 0
        self.label_pad_token_id = -100

        # Task type validation
        self.task_type = self.config["task_type"].lower()
        assert self.task_type in TASK_TYPES, f"{self.task_type} is not implemented!"

        # Token components detection
        self.token_components = None
        self.all_token_ids = None

    def _load_vocab(self):
        fn = os.path.join(
            self.config["name_or_path"], self.config.get("vocab_file", "vocab")
        )
        return load_vocab(fn)

    def get_vocab_size(self):
        return self.vocab_size

    def get_all_vocab_ids(self):
        if self.all_token_ids is None:
            self.all_token_ids = tuple(range(self.vocab_size))
        return self.all_token_ids

    # Token getters (moved from GSTTokenizer)
    def get_bos_token(self):
        return self.config["structure"]["node"]["bos_token"]

    def get_eos_token(self):
        return self.config["structure"]["node"]["eos_token"]

    def get_new_node_token(self):
        return self.config["structure"]["node"].get(
            "new_node_token", self.get_label_pad_token()
        )

    def get_edge_bi_token(self):
        return self.config["structure"]["edge"]["bi_token"]

    def get_jump_token(self):
        return self.config["structure"]["edge"]["jump_token"]

    def get_gsum_token(self):
        return self.config["structure"]["graph"]["summary_token"]

    def get_mask_token(self):
        return self.config["structure"]["common"].get("mask_token", "<mask>")

    def get_icl_token(self):
        return self.config["structure"]["common"]["icl_token"]

    def get_sep_token(self):
        return self.config["structure"]["common"]["sep_token"]

    def get_common_structure(self):
        return self.config["structure"]["common"].get("reserved_token", [])

    def get_common_semantics(self):
        return self.config["semantics"]["common"].get("reserved_token", [])

    def get_label_pad_token(self):
        return "<label_pad>"

    def get_bos_token_id(self):
        return self.vocab_map[self.get_bos_token()]

    def get_eos_token_id(self):
        return self.vocab_map[self.get_eos_token()]

    def get_new_node_token_id(self):
        return self.vocab_map[self.get_new_node_token()]

    def get_jump_token_id(self):
        return self.vocab_map[self.get_jump_token()]

    def get_gsum_token_id(self):
        return self.vocab_map.get(self.get_gsum_token(), None)

    def get_mask_token_id(self):
        return self.vocab_map[self.get_mask_token()]

    @abstractmethod
    def tokenize(self, graph: Data) -> TokenizationOutput:
        """Convert graph to token sequence."""
        pass

    @abstractmethod
    def convert_tokens_to_ids(self, seq_tokens, seq_labels) -> Dict:
        """Convert token sequences to IDs."""
        pass

    def __call__(self, graph: Data, is_training: Optional[bool] = None):
        """
        Full tokenization pipeline:
        1. Tokenize graph
        2. Pack sequences (if sequence_packer is set)
        3. Convert to IDs
        4. Prepare for task
        """
        # 1. Tokenize
        token_res = self.tokenize(graph)

        # 2. Pack (optional)
        if self.sequence_packer is not None:
            ls_tokens, ls_labels, ls_embed, ls_len = self.sequence_packer.pack(
                token_res, graph.idx, self.tokenize
            )
            token_res.ls_tokens = ls_tokens
            token_res.ls_labels = ls_labels
            token_res.ls_embed = ls_embed
            token_res.ls_len = ls_len
        else:
            token_res.ls_len = [len(token_res.ls_tokens)]

        # 3. Convert to IDs
        in_dict = self.convert_tokens_to_ids(token_res.ls_tokens, token_res.ls_labels)
        if token_res.ls_embed:
            in_dict["embed"] = token_res.ls_embed

        # 4. Prepare for task
        if self.task_preparer is not None:
            in_dict = self.task_preparer.prepare(in_dict, token_res, graph, self)

        return in_dict

    def pad(self, features, **kwargs):
        """Delegate to padding strategy."""
        if self.padding_strategy is None:
            raise RuntimeError("No padding strategy configured")
        return self.padding_strategy.pad_batch(features, **kwargs)

    def _map_tokens_to_ids(self, tokens):
        """Helper to map tokens to IDs."""
        if tokens is None:
            return None
        elif isinstance(tokens, str):
            return self.vocab_map[tokens]
        elif isinstance(tokens, (list, tuple)):
            return [self.vocab_map[t] for t in tokens]
        else:
            raise ValueError(f"Unsupported token type: {type(tokens)}")
