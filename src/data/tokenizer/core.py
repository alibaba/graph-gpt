"""Core tokenizer implementations."""

from typing import Dict, List
import torch
from torch_geometric.data import Data

from .base import BaseTokenizer
from .types import TokenizationOutput
from .padding import get_input_dict_from_seq_tokens_id
from ...conf import TrainingConfig


class GSTTokenizer(BaseTokenizer):
    """
    Graph Sequence Tokenizer for 1D token sequences.

    Suitable for:
    - Pre-training with MLM
    - Node-level tasks
    - Edge-level tasks
    """

    def __init__(
        self,
        config: Dict,
        *,
        padding_side: str = "right",
        add_eos: bool = True,
        train_cfg: TrainingConfig = None,
        **kwargs,
    ):
        # Import strategies here to avoid circular imports
        from .strategies.padding import FlatPaddingStrategy
        from .strategies.task_prep import get_task_strategy

        # Create padding strategy
        padding_strategy = FlatPaddingStrategy(
            pad_token_id=0,
            label_pad_token_id=-100,
            padding_side=padding_side,
        )

        # Create task preparation strategy
        task_type = config["task_type"].lower()
        task_preparer = get_task_strategy(task_type)

        super().__init__(
            config,
            padding_strategy=padding_strategy,
            task_preparer=task_preparer,
            add_eos=add_eos,
            train_cfg=train_cfg,
            **kwargs,
        )

        self.mask_type = self.config["semantics"].get("attr_assignment", "random")
        self.label_to_be_padded = self._get_label_token_id_to_be_padded()

        # For sequence packing (optional, set externally if needed)
        self.mpe = None
        self.dataset = None
        self.sampler = None
        self.random_ratio = 1.0

    def _get_label_token_id_to_be_padded(self):
        """Get label token IDs that should be padded."""
        if self.task_type != "pretrain":
            return set()
        label_tokens_to_be_padded = set(self.config.get("label_tokens_to_pad", []))
        label_token_ids = [self.vocab_map[token] for token in label_tokens_to_be_padded]
        return set(label_token_ids)

    def get_token_components(self, ls_tokens):
        """Determine token components (0 for 1D tokens)."""
        if self.token_components is None:
            if ls_tokens and isinstance(ls_tokens[0], list):
                self.token_components = len(ls_tokens[0])
            else:
                self.token_components = 0
        return self.token_components

    def setup_sequence_packing(self, mpe, dataset, sampler=None, random_ratio=1.0):
        """Setup sequence packing for pre-training."""
        from .strategies.packing import SequencePacker

        self.mpe = mpe
        self.dataset = dataset
        self.sampler = sampler
        self.random_ratio = random_ratio

        self.sequence_packer = SequencePacker(
            mpe=mpe,
            dataset=dataset,
            sampler=sampler,
            random_ratio=random_ratio,
            eos_token=self.get_eos_token(),
            label_pad_token=self.get_label_pad_token(),
        )

    def tokenize(self, graph: Data) -> TokenizationOutput:
        """Tokenize graph to 1D token sequence."""
        from ...utils import nx_utils, instruct_tuning_utils, graph2path
        from .masking import get_mask_of_raw_seq
        from .graph_encoding import get_semantics_raw_node_edge2attr_mapping

        # 1-2. Transform graph to Eulerian sequence
        assert (
            graph.num_nodes <= self.config["structure"]["node"]["node_scope"]
        ), f"num_nodes: {graph.num_nodes} > node_scope: {self.config['structure']['node']['node_scope']}"

        path = graph2path(graph)

        # 3. Obtain mappings
        node_structure_mapping = nx_utils.get_structure_raw_node2idx_mapping(
            path,
            self.config["structure"]["node"]["scope_base"],
            self.config["structure"]["node"]["node_scope"],
            self.config["structure"]["node"].get("cyclic", False),
        )
        edge_structure_mapping = nx_utils.get_structure_raw_edge2type_mapping(
            path, graph
        )

        (
            node_semantics_mapping,
            edge_semantics_mapping,
            graph_semantics_mapping,
        ) = get_semantics_raw_node_edge2attr_mapping(path, graph, self.config)

        # 3.1 Target node/edge tokens
        tgt_node_token = None
        tgt_edge_src_token = None
        tgt_edge_dst_token = None
        if hasattr(graph, "root_n_id"):
            if isinstance(graph.root_n_id, int):
                tgt_node_token = node_structure_mapping[graph.root_n_id]
            elif (
                isinstance(graph.root_n_id, torch.Tensor) and len(graph.root_n_id) == 2
            ):
                src, dst = graph.root_n_id.tolist()
                tgt_edge_src_token = node_structure_mapping[src]
                tgt_edge_dst_token = node_structure_mapping[dst]

        # 4. Decorate and get raw sequence
        raw_seq = nx_utils.get_raw_seq_from_path(path)
        mask = get_mask_of_raw_seq(raw_seq, self.mask_type)

        ls_tokens, _, _ = nx_utils.decorate_node_edge_graph_with_mask(
            self,
            raw_seq,
            mask,
            node_structure_mapping,
            edge_structure_mapping,
            node_semantics_mapping,
            edge_semantics_mapping,
            graph_semantics_mapping,
            attr_shuffle=self.config["semantics"].get("attr_shuffle", False),
        )

        # 5. Remove bidirectional edge-type tokens if configured
        dict_edge = self.config["structure"]["edge"]
        if dict_edge.get("remove_edge_type_token", False):
            edge_types = {dict_edge["bi_token"]}
            ls_tokens = [t for t in ls_tokens if t not in edge_types]

        # 5.1 Get labels from input tokens
        ls_labels = nx_utils.get_labels_from_input_tokens(ls_tokens, self)

        # 6. Add special tokens
        ls_struct_tokens, ls_struct_labels = nx_utils.understand_structure(
            graph, self.config, node_structure_mapping, edge_structure_mapping, self
        )
        ls_tokens.extend(ls_struct_tokens)
        ls_labels.extend(ls_struct_labels)

        (
            ls_instruct_tokens,
            ls_instruct_labels,
        ) = instruct_tuning_utils.follow_instructions(
            graph,
            self.config,
            node_structure_mapping,
            edge_structure_mapping,
            node_semantics_mapping,
            edge_semantics_mapping,
            self,
        )
        ls_tokens.extend(ls_instruct_tokens)
        ls_labels.extend(ls_instruct_labels)

        # 6.3 Add EOS
        if self.add_eos and "pretrain" not in self.task_type:
            ls_tokens.append(self.get_eos_token())
            ls_labels.append(self.get_eos_token())

        return TokenizationOutput(
            ls_tokens=ls_tokens,
            ls_labels=ls_labels,
            tgt_node_token=tgt_node_token,
            tgt_edge_src_token=tgt_edge_src_token,
            tgt_edge_dst_token=tgt_edge_dst_token,
        )

    def convert_tokens_to_ids(self, seq_tokens, seq_labels) -> Dict:
        """Convert 1D token sequences to IDs."""
        seq_tokens_id = [self.vocab_map[token] for token in seq_tokens]
        seq_labels_id = [self.vocab_map[token] for token in seq_labels]

        return get_input_dict_from_seq_tokens_id(
            seq_tokens_id,
            seq_labels_id,
            self.label_to_be_padded,
            self.label_pad_token_id,
        )


class StackedGSTTokenizer(BaseTokenizer):
    """
    Stacked Graph Sequence Tokenizer for 2D token sequences.

    Suitable for:
    - Graph-level tasks with stacked node/edge attributes
    - Short and Long stacking methods
    """

    # Rotation functions are initialized lazily in __init__
    DICT_pos_func = {"trans_rotate": None, "anchor_rotate": None}

    def __init__(
        self,
        config: Dict,
        *,
        padding_side: str = "right",
        add_eos: bool = True,
        train_cfg: TrainingConfig = None,
        stack_method: str = "short",
        rotation: str = "anchor_rotate",
        **kwargs,
    ):
        from .strategies.padding import StackedPaddingStrategy
        from .strategies.task_prep import get_task_strategy
        from ...utils import mol_utils

        # Initialize rotation functions
        self.DICT_pos_func = {
            "trans_rotate": mol_utils.trans_rotate_3d_random,
            "anchor_rotate": mol_utils.rotate_3d_v3,
        }

        assert stack_method in {"short", "long"}
        self.stack_method = stack_method

        # Create padding strategy for 2D sequences
        padding_strategy = StackedPaddingStrategy(
            pad_token_id=0,
            label_pad_token_id=-100,
            padding_side=padding_side,
        )

        # Create task preparation strategy
        task_type = config["task_type"].lower()
        task_preparer = get_task_strategy(task_type)

        super().__init__(
            config,
            padding_strategy=padding_strategy,
            task_preparer=task_preparer,
            add_eos=add_eos,
            train_cfg=train_cfg,
            **kwargs,
        )

        self.rotation = rotation
        self.default_node_attr = None
        self.default_edge_attr = None
        self.default_node_embed = None
        self.default_edge_embed = None
        self.default_edge_attr_id = None

        # Set ignored values to None for stacking
        self.config["semantics"]["node"]["ignored_val"] = None
        self.config["semantics"]["edge"]["ignored_val"] = None

    def get_default_node_attr(self, graph=None):
        if self.default_node_attr is None:
            from .stacking import get_default_semantics_attr_mapping

            self.default_node_attr = get_default_semantics_attr_mapping(
                graph, self.config, "node"
            )
        return self.default_node_attr

    def get_default_edge_attr(self, graph=None):
        if self.default_edge_attr is None:
            from .stacking import get_default_semantics_attr_mapping

            self.default_edge_attr = get_default_semantics_attr_mapping(
                graph, self.config, "edge"
            )
        return self.default_edge_attr

    def get_default_edge_attr_id(self, graph=None):
        if self.default_edge_attr_id is None:
            default_edge_attr = self.get_default_edge_attr(graph)
            self.default_edge_attr_id = self._map_tokens_to_ids(default_edge_attr)
        return self.default_edge_attr_id

    def get_default_node_embed(self, graph=None):
        if self.default_node_embed is None:
            from .stacking import get_default_semantics_embed_mapping

            self.default_node_embed = get_default_semantics_embed_mapping(
                graph, self.config, "node"
            )
        return self.default_node_embed

    def get_default_edge_embed(self, graph=None):
        if self.default_edge_embed is None:
            from .stacking import get_default_semantics_embed_mapping

            self.default_edge_embed = get_default_semantics_embed_mapping(
                graph, self.config, "edge"
            )
        return self.default_edge_embed

    def get_token_components(self, ls_tokens):
        """Determine token components for 2D tokens."""
        if self.token_components is None:
            if ls_tokens and isinstance(ls_tokens[0], list):
                self.token_components = len(ls_tokens[0])
            else:
                self.token_components = 0
        return self.token_components

    def setup_sequence_packing(self, mpe, dataset, sampler=None, random_ratio=1.0):
        """Setup sequence packing for pre-training."""
        from .strategies.packing import SequencePacker

        self.mpe = mpe
        self.dataset = dataset
        self.sampler = sampler
        self.random_ratio = random_ratio

        self.sequence_packer = SequencePacker(
            mpe=mpe,
            dataset=dataset,
            sampler=sampler,
            random_ratio=random_ratio,
            eos_token=self.get_eos_token(),
            label_pad_token=self.get_label_pad_token(),
        )

    def tokenize(self, graph: Data) -> TokenizationOutput:
        """Tokenize graph to 2D stacked token sequence."""
        from ...utils import nx_utils, instruct_tuning_utils, graph2path
        from .stacking import (
            add_eos_embed,
            stack_node_edge_graph_attr_to_node,
            stack_attr_to_node_and_edge,
            get_default_semantics_embed_mapping,
        )
        from .graph_encoding import get_semantics_raw_node_edge2attr_mapping

        # Apply rotation if positions exist
        if hasattr(graph, "pos") and graph.pos is not None:
            graph.pos = self.DICT_pos_func[self.rotation](graph.pos)
        if hasattr(graph, "rdkit_pos"):
            from ...utils import mol_utils

            graph.rdkit_pos = mol_utils.rotate_3d_v3(graph.rdkit_pos)
            graph.pos = torch.hstack([graph.pos, graph.rdkit_pos])

        # 1-2. Get Eulerian path
        path = graph2path(graph, prioritize=self.task_type != "pretrain")

        # 3. Obtain mappings
        node_structure_mapping = nx_utils.get_structure_raw_node2idx_mapping(
            path,
            self.config["structure"]["node"]["scope_base"],
            self.config["structure"]["node"]["node_scope"],
            self.config["structure"]["node"].get("cyclic", False),
        )
        edge_structure_mapping = nx_utils.get_structure_raw_edge2type_mapping(
            path, graph
        )

        (
            node_semantics_mapping,
            edge_semantics_mapping,
            graph_semantics_mapping,
        ) = get_semantics_raw_node_edge2attr_mapping(path, graph, self.config)

        # Add default values for compatibility
        node_structure_mapping[-1] = (self.get_new_node_token(),)
        edge_structure_mapping[(-1, -1)] = self.get_edge_bi_token()

        if node_semantics_mapping["discrete"]:
            node_semantics_mapping["discrete"][-1] = self.get_default_node_attr(graph)
        if edge_semantics_mapping["discrete"]:
            edge_semantics_mapping["discrete"][(-1, -1)] = self.get_default_edge_attr(
                graph
            )
        if node_semantics_mapping["embed"]:
            node_semantics_mapping["embed"][-1] = self.get_default_node_embed(graph)
        if edge_semantics_mapping["embed"]:
            edge_semantics_mapping["embed"][(-1, -1)] = self.get_default_edge_embed(
                graph
            )

        # 3.1 Target tokens
        tgt_node_token = None
        tgt_edge_src_token = None
        tgt_edge_dst_token = None
        tgt_edge_attr_token = None
        if hasattr(graph, "root_n_id"):
            if isinstance(graph.root_n_id, int):
                tgt_node_token = node_structure_mapping[graph.root_n_id]
            elif (
                isinstance(graph.root_n_id, torch.Tensor) and len(graph.root_n_id) == 2
            ):
                src, dst = graph.root_n_id.tolist()
                tgt_edge_src_token = node_structure_mapping[src]
                tgt_edge_dst_token = node_structure_mapping[dst]
                if hasattr(graph, "tgt_edge_attr"):
                    tgt_edge_attr_token = self._get_tokens_from_single_edge_attr(
                        graph.tgt_edge_attr
                    )

        # Handle nodev2 task type
        if self.task_type == "nodev2":
            assert tgt_node_token is None
            tgt_node_token = self._flatten_list(
                [node_structure_mapping[ele] for ele in range(graph.num_nodes)]
            )

        # 4. Remove edge type tokens if configured
        if self.config["structure"]["edge"]["remove_edge_type_token"]:
            edge_structure_mapping = None

        # 5. Stack attributes
        stack_func = (
            stack_node_edge_graph_attr_to_node
            if self.stack_method == "short"
            else stack_attr_to_node_and_edge
        )
        ls_tokens, ls_embed, ls_raw_node_idx = stack_func(
            self,
            path,
            node_structure_mapping,
            edge_structure_mapping,
            node_semantics_mapping,
            edge_semantics_mapping,
            graph_semantics_mapping,
        )

        # 6. Add EOS
        if self.add_eos:
            eos_token = self.config["structure"]["node"]["eos_token"]
            token_components = len(ls_tokens[0])
            ls_tokens.append([eos_token] * token_components)
            ls_embed = add_eos_embed(ls_embed)
            ls_raw_node_idx.append(-1)

        token_components = len(ls_tokens[0])
        ls_labels = ls_tokens[1:] + [[self.get_eos_token()] * token_components]

        # 6.2 Instruction tuning
        (
            ls_instruct_tokens,
            ls_instruct_labels,
        ) = instruct_tuning_utils.follow_instructions(
            graph,
            self.config,
            node_structure_mapping,
            edge_structure_mapping,
            node_semantics_mapping,
            edge_semantics_mapping,
            self,
        )
        if ls_instruct_tokens:
            ls_tokens.extend(ls_instruct_tokens)
            ls_labels.extend(ls_instruct_labels)

        if ls_embed and ls_instruct_tokens:
            assert (
                False
            ), "NOT implemented when embed inputs is presented with instructions"

        return TokenizationOutput(
            ls_tokens=ls_tokens,
            ls_labels=ls_labels,
            tgt_node_token=tgt_node_token,
            tgt_edge_src_token=tgt_edge_src_token,
            tgt_edge_dst_token=tgt_edge_dst_token,
            tgt_edge_attr_token=tgt_edge_attr_token,
            ls_embed=ls_embed,
            ls_raw_node_idx=ls_raw_node_idx,
        )

    def convert_tokens_to_ids(self, seq_tokens: List[List[str]], seq_labels) -> Dict:
        """Convert 2D stacked token sequences to IDs."""
        seq_tokens_id = [
            [self.vocab_map[token] for token in feat_tokens]
            for feat_tokens in seq_tokens
        ]
        seq_labels_id = [
            [self.vocab_map[token] for token in feat_tokens]
            for feat_tokens in seq_labels
        ]

        return get_input_dict_from_seq_tokens_id(
            seq_tokens_id, seq_labels_id, set(), None
        )

    def _get_tokens_from_single_edge_attr(self, edge_attr: torch.Tensor):
        """Get tokens from a single edge attribute tensor."""
        from .graph_encoding import _tokenize_discrete_attr

        assert len(edge_attr.shape) == 1
        tokens = []
        v = edge_attr.numpy()
        node_or_edge = "edge"
        discrete_attr = self.config["semantics"][node_or_edge]["discrete"]
        world_identifier = self.config["attr_world_identifier"]

        if discrete_attr is not None:
            share_vocab = self.config["semantics"][node_or_edge].get(
                "share_vocab", False
            )
            ignored_val = self.config["semantics"][node_or_edge]["ignored_val"]
            tokens = _tokenize_discrete_attr(
                v.astype(str),
                world_identifier,
                node_or_edge,
                ignored_val=ignored_val,
                shuffle=False,
                share_vocab=share_vocab,
            )
        return tokens

    @staticmethod
    def _flatten_list(nested_list):
        """Flatten a nested list."""
        return [item for sublist in nested_list for item in sublist]
