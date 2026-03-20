"""
Smoke tests for tokenizer public API surface.

These tests verify that all public imports, class instantiation patterns,
and key utility functions work correctly. They require no GPU or real datasets.
"""

import sys
import os
import pytest
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# 1. Import-level tests: verify that all public import paths resolve
# ---------------------------------------------------------------------------


class TestImports:
    """Verify every externally-used import path still resolves."""

    def test_import_tokenizer_classes_from_package(self):
        from src.data.tokenizer import GSTTokenizer, StackedGSTTokenizer

        assert GSTTokenizer is not None
        assert StackedGSTTokenizer is not None

    def test_import_tokenizer_module_and_getattr(self):
        """The training modes use `from ..data import tokenizer`
        then `getattr(tokenizer, tokenizer_config['tokenizer_class'])`."""
        from src.data import tokenizer

        cls = getattr(tokenizer, "GSTTokenizer")
        assert cls is not None
        cls2 = getattr(tokenizer, "StackedGSTTokenizer")
        assert cls2 is not None

    def test_import_collator(self):
        from src.data.collator import DataCollatorForGST

        assert DataCollatorForGST is not None

    def test_import_prepare_inputs_from_utils(self):
        from src.utils import prepare_inputs_for_task

        assert callable(prepare_inputs_for_task)

    def test_import_constants_from_tokenizer_utils(self):
        from src.utils.tokenizer_utils import MOL_ENERGY_BIN_LEN, MOL_ENERGY_SCALE

        assert MOL_ENERGY_BIN_LEN == 16
        assert MOL_ENERGY_SCALE == 1000

    def test_import_tokenization_output(self):
        from src.utils.tokenizer_utils import TokenizationOutput

        assert TokenizationOutput is not None

    def test_import_vocab_builder(self):
        from src.data.vocab_builder import load_vocab, build_vocab

        assert callable(load_vocab)
        assert callable(build_vocab)

    def test_import_get_inputs_preparation_func(self):
        from src.utils.tokenizer_utils import get_inputs_preparation_func

        assert callable(get_inputs_preparation_func)

    def test_import_mask_of_raw_seq(self):
        from src.utils.tokenizer_utils import get_mask_of_raw_seq

        assert callable(get_mask_of_raw_seq)


# ---------------------------------------------------------------------------
# 2. Lightweight unit tests for key utility functions
# ---------------------------------------------------------------------------


class TestTokenizationOutput:
    """Verify TokenizationOutput can be instantiated and accessed."""

    def test_create_default(self):
        from src.utils.tokenizer_utils import TokenizationOutput

        out = TokenizationOutput()
        assert out.ls_tokens is None
        assert out.ls_labels is None

    def test_create_with_values(self):
        from src.utils.tokenizer_utils import TokenizationOutput

        out = TokenizationOutput(
            ls_tokens=["a", "b", "c"],
            ls_labels=["b", "c", "<eos>"],
        )
        assert out.ls_tokens == ["a", "b", "c"]
        assert out.ls_labels == ["b", "c", "<eos>"]

    def test_field_mutation(self):
        from src.utils.tokenizer_utils import TokenizationOutput

        out = TokenizationOutput(ls_tokens=["a"], ls_labels=["b"])
        out.ls_tokens = ["x", "y"]
        assert out.ls_tokens == ["x", "y"]


class TestGetMaskOfRawSeq:
    """Verify get_mask_of_raw_seq produces expected shapes and values."""

    def test_basic_mask_first(self):
        from src.utils.tokenizer_utils import get_mask_of_raw_seq

        raw_seq = [0, (0, 1), 1, (1, 2), 2]
        mask = get_mask_of_raw_seq(raw_seq, mask_type="first")
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (5,)
        assert mask.sum() > 0

    def test_mask_all(self):
        from src.utils.tokenizer_utils import get_mask_of_raw_seq

        raw_seq = [0, (0, 1), 1, (1, 0), 0]
        mask = get_mask_of_raw_seq(raw_seq, mask_type="all")
        assert mask.shape == (5,)
        assert mask.sum() == 5  # all positions masked

    def test_single_element(self):
        from src.utils.tokenizer_utils import get_mask_of_raw_seq

        raw_seq = [42]
        mask = get_mask_of_raw_seq(raw_seq, mask_type="first")
        assert mask.shape == (1,)
        assert mask[0] == 1


class TestGetInputDictFromSeqTokensId:
    """Verify get_input_dict_from_seq_tokens_id returns correct dict structure."""

    def test_basic_with_labels(self):
        from src.data.tokenizer import get_input_dict_from_seq_tokens_id

        seq_tokens_id = [1, 2, 3, 4]
        seq_labels_id = [2, 3, 4, 5]
        result = get_input_dict_from_seq_tokens_id(
            seq_tokens_id, seq_labels_id, set(), -100
        )
        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result
        assert len(result["input_ids"]) == 4
        assert len(result["labels"]) == 4
        assert result["attention_mask"] == [1, 1, 1, 1]

    def test_with_label_padding(self):
        from src.data.tokenizer import get_input_dict_from_seq_tokens_id

        seq_tokens_id = [1, 2, 3]
        seq_labels_id = [2, 3, 4]
        label_to_be_pad = {3}
        result = get_input_dict_from_seq_tokens_id(
            seq_tokens_id, seq_labels_id, label_to_be_pad, -100
        )
        assert result["labels"][1] == -100  # token_id 3 should be padded

    def test_autoregressive_mode(self):
        from src.data.tokenizer import get_input_dict_from_seq_tokens_id

        seq_tokens_id = [1, 2, 3, 4, 5]
        result = get_input_dict_from_seq_tokens_id(
            seq_tokens_id, None, set(), -100
        )
        assert result["input_ids"] == [1, 2, 3, 4]
        assert result["labels"] == [2, 3, 4, 5]


class TestPrepareInputsRegistration:
    """Verify that task-type registration works for key task types."""

    def test_known_task_types(self):
        from src.utils.tokenizer_utils import get_inputs_preparation_func

        known_types = [
            "pretrain",
            "pretrain-mlm",
            "pretrain-coord",
            "graph",
            "edge",
            "node",
            "nodev2",
        ]
        for task_type in known_types:
            func = get_inputs_preparation_func(task_type)
            assert func is not None, f"No registered function for task type '{task_type}'"
            assert callable(func)

    def test_unknown_task_type_returns_none(self):
        from src.utils.tokenizer_utils import get_inputs_preparation_func

        func = get_inputs_preparation_func("nonexistent_task_type_xyz")
        assert func is None


# ---------------------------------------------------------------------------
# 3. Module-level function accessibility
# ---------------------------------------------------------------------------


class TestModuleLevelFunctions:
    """Verify key module-level functions are accessible from their expected locations."""

    def test_dict_pos_func(self):
        from src.data.tokenizer import DICT_pos_func

        assert isinstance(DICT_pos_func, dict)
        assert "trans_rotate" in DICT_pos_func
        assert "anchor_rotate" in DICT_pos_func

    def test_get_semantics_raw_node_edge2attr_mapping(self):
        from src.data.tokenizer import get_semantics_raw_node_edge2attr_mapping

        assert callable(get_semantics_raw_node_edge2attr_mapping)

    def test_mask_semantics_raw_node_edge_attr(self):
        from src.data.tokenizer import mask_semantics_raw_node_edge_attr

        assert callable(mask_semantics_raw_node_edge_attr)
