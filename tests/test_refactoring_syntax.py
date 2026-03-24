"""
Syntax and import tests for the refactored tokenizer.
These tests verify the code structure without requiring all dependencies.
"""

import ast
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSyntaxValidity:
    """Verify all refactored files have valid Python syntax."""

    def test_base_tokenizer_syntax(self):
        with open("src/data/tokenizer/base.py") as f:
            code = f.read()
        ast.parse(code)  # Raises SyntaxError if invalid

    def test_core_tokenizer_syntax(self):
        with open("src/data/tokenizer/core.py") as f:
            code = f.read()
        ast.parse(code)

    def test_padding_strategy_syntax(self):
        with open("src/data/tokenizer/strategies/padding.py") as f:
            code = f.read()
        ast.parse(code)

    def test_packing_strategy_syntax(self):
        with open("src/data/tokenizer/strategies/packing.py") as f:
            code = f.read()
        ast.parse(code)

    def test_task_prep_base_syntax(self):
        with open("src/data/tokenizer/strategies/task_prep/base.py") as f:
            code = f.read()
        ast.parse(code)

    def test_task_prep_pretrain_syntax(self):
        with open("src/data/tokenizer/strategies/task_prep/pretrain.py") as f:
            code = f.read()
        ast.parse(code)

    def test_task_prep_supervised_syntax(self):
        with open("src/data/tokenizer/strategies/task_prep/supervised.py") as f:
            code = f.read()
        ast.parse(code)

    def test_strategies_init_syntax(self):
        with open("src/data/tokenizer/strategies/__init__.py") as f:
            code = f.read()
        ast.parse(code)

    def test_task_prep_init_syntax(self):
        with open("src/data/tokenizer/strategies/task_prep/__init__.py") as f:
            code = f.read()
        ast.parse(code)


class TestClassStructure:
    """Verify the class hierarchy is correctly defined."""

    def test_base_tokenizer_is_abstract(self):
        with open("src/data/tokenizer/base.py") as f:
            tree = ast.parse(f.read())

        found_base = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "BaseTokenizer":
                found_base = True
                # Check it inherits from ABC
                bases = [isinstance(b, ast.Name) and b.id for b in node.bases]
                assert "ABC" in bases, "BaseTokenizer should inherit from ABC"

                # Check for abstract methods
                has_abstract_method = False
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        for decorator in item.decorator_list:
                            if (
                                isinstance(decorator, ast.Name)
                                and decorator.id == "abstractmethod"
                            ):
                                has_abstract_method = True
                                break
                assert has_abstract_method, "BaseTokenizer should have abstract methods"

        assert found_base, "BaseTokenizer class not found"

    def test_gst_tokenizer_inherits_base(self):
        with open("src/data/tokenizer/core.py") as f:
            tree = ast.parse(f.read())

        found_gst = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "GSTTokenizer":
                found_gst = True
                bases = [isinstance(b, ast.Name) and b.id for b in node.bases]
                assert (
                    "BaseTokenizer" in bases
                ), "GSTTokenizer should inherit from BaseTokenizer"

        assert found_gst, "GSTTokenizer class not found"

    def test_stacked_gst_tokenizer_inherits_base(self):
        with open("src/data/tokenizer/core.py") as f:
            tree = ast.parse(f.read())

        found_stacked = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "StackedGSTTokenizer":
                found_stacked = True
                bases = [isinstance(b, ast.Name) and b.id for b in node.bases]
                assert (
                    "BaseTokenizer" in bases
                ), "StackedGSTTokenizer should inherit from BaseTokenizer"

        assert found_stacked, "StackedGSTTokenizer class not found"

    def test_padding_strategy_is_abstract(self):
        with open("src/data/tokenizer/strategies/padding.py") as f:
            tree = ast.parse(f.read())

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "PaddingStrategy":
                found = True
                bases = [isinstance(b, ast.Name) and b.id for b in node.bases]
                assert "ABC" in bases, "PaddingStrategy should inherit from ABC"

        assert found, "PaddingStrategy class not found"

    def test_flat_padding_inherits_padding(self):
        with open("src/data/tokenizer/strategies/padding.py") as f:
            tree = ast.parse(f.read())

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "FlatPaddingStrategy":
                found = True
                bases = [isinstance(b, ast.Name) and b.id for b in node.bases]
                assert (
                    "PaddingStrategy" in bases
                ), "FlatPaddingStrategy should inherit from PaddingStrategy"

        assert found, "FlatPaddingStrategy class not found"

    def test_stacked_padding_inherits_padding(self):
        with open("src/data/tokenizer/strategies/padding.py") as f:
            tree = ast.parse(f.read())

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "StackedPaddingStrategy":
                found = True
                bases = [isinstance(b, ast.Name) and b.id for b in node.bases]
                assert (
                    "PaddingStrategy" in bases
                ), "StackedPaddingStrategy should inherit from PaddingStrategy"

        assert found, "StackedPaddingStrategy class not found"

    def test_task_prep_strategy_is_abstract(self):
        with open("src/data/tokenizer/strategies/task_prep/base.py") as f:
            tree = ast.parse(f.read())

        found = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "TaskPreparationStrategy"
            ):
                found = True
                bases = [isinstance(b, ast.Name) and b.id for b in node.bases]
                assert "ABC" in bases, "TaskPreparationStrategy should inherit from ABC"

        assert found, "TaskPreparationStrategy class not found"

    def test_task_strategy_map_exists(self):
        with open("src/data/tokenizer/strategies/task_prep/__init__.py") as f:
            tree = ast.parse(f.read())

        found_map = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "TASK_STRATEGY_MAP"
                    ):
                        found_map = True

        assert found_map, "TASK_STRATEGY_MAP not found"


class TestCompositionPattern:
    """Verify composition is used instead of inheritance where appropriate."""

    def test_base_tokenizer_has_padding_strategy_attr(self):
        with open("src/data/tokenizer/base.py") as f:
            content = f.read()

        assert (
            "padding_strategy" in content
        ), "BaseTokenizer should have padding_strategy attribute"

    def test_base_tokenizer_has_task_preparer_attr(self):
        with open("src/data/tokenizer/base.py") as f:
            content = f.read()

        assert (
            "task_preparer" in content
        ), "BaseTokenizer should have task_preparer attribute"

    def test_base_tokenizer_has_sequence_packer_attr(self):
        with open("src/data/tokenizer/base.py") as f:
            content = f.read()

        assert (
            "sequence_packer" in content
        ), "BaseTokenizer should have sequence_packer attribute"


class TestBackwardCompatibility:
    """Verify backward compatibility exports."""

    def test_legacy_exports_exist(self):
        with open("src/data/tokenizer/__init__.py") as f:
            content = f.read()

        # Check that legacy names are in __all__
        legacy_names = [
            "DICT_pos_func",
            "get_semantics_raw_node_edge2attr_mapping",
            "_tokenize_discrete_attr",
            "_merge_two_ls",
        ]
        for name in legacy_names:
            assert name in content, f"Legacy export {name} should be in __init__.py"

    def test_gst_tokenizer_in_exports(self):
        with open("src/data/tokenizer/__init__.py") as f:
            content = f.read()

        assert "GSTTokenizer" in content
        assert "StackedGSTTokenizer" in content

    def test_new_strategies_in_exports(self):
        with open("src/data/tokenizer/__init__.py") as f:
            content = f.read()

        new_exports = [
            "BaseTokenizer",
            "PaddingStrategy",
            "FlatPaddingStrategy",
            "StackedPaddingStrategy",
            "SequencePacker",
            "TaskPreparationStrategy",
        ]
        for name in new_exports:
            assert name in content, f"New export {name} should be in __init__.py"
