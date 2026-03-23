import os
import random
import math
import numpy as np
from typing import Dict, List, Callable, Tuple, Optional, Union, Iterable, Set

import torch
from torch_geometric.data import Data
from torch.utils.data import IterableDataset, Dataset
from ...utils import (
    nx_utils,
    instruct_tuning_utils,
    mol_utils,
    attn_mask_utils,
    graph2path,
)
from ...conf import TASK_TYPES
from ...conf import TrainingConfig
from ..vocab_builder import load_vocab
from .types import TokenizationOutput
from .masking import get_mask_of_raw_seq, _pad_stacked_targets
from .task_prep import prepare_inputs_for_task
from .graph_encoding import (
    _tokenize_discrete_attr,
    _tokenize_continuous_attr,
    _remove_lead_zero,
    _add_regression_token,
    _get_node2attr_mapping,
    _get_edge2attr_mapping,
    _get_graph2attr_mapping,
    get_semantics_attr_mapping,
    get_semantics_raw_node_edge2attr_mapping,
    mask_semantics_attr,
    mask_semantics_raw_node_edge_attr,
)
from .stacking import (
    add_eos_embed,
    stack_node_edge_graph_attr_to_node,
    stack_attr_to_node_and_edge,
    get_default_semantics_attr_mapping,
    get_default_semantics_embed_mapping,
)
from .padding import (
    _merge_two_ls,
    _get_batch_seq_len,
    get_input_dict_from_seq_tokens_id,
)


DICT_pos_func = {
    "trans_rotate": mol_utils.trans_rotate_3d_random,
    "anchor_rotate": mol_utils.rotate_3d_v3,
}


class GSTTokenizer(object):
    def __init__(
        self,
        config: Dict,
        *,
        padding_side: str = "right",
        add_eos: bool = True,
        train_cfg: TrainingConfig = None,
        **kwargs,
    ):
        self.config = config
        self.mask_type = self.config["semantics"].get("attr_assignment", "random")
        assert padding_side in {"left", "right"}
        self.padding_side = padding_side
        self.add_eos = add_eos
        self.vocab_map = self.load_vocab()
        self.vocab_size = self.get_vocab_size()
        self.label_pad_token_id = -100
        self.pad_token_id = 0
        self.task_type = self.config["task_type"].lower()
        assert self.task_type in TASK_TYPES, f"{self.task_type} is not implemented!"
        self.semantics2tokens_mapping = get_semantics_raw_node_edge2attr_mapping
        # below for pack target token sequence with randomly sampled token sequence
        self.mpe = None
        self.dataset = None
        self.sampler = None
        self.token_components = None
        self.random_ratio = 1
        self.label_to_be_padded = self.get_label_token_id_to_be_padded()
        self.node_idx_tokens = None
        self.node_idx_token_ids = None
        self.all_token_ids = None
        # kwargs
        self.train_cfg = train_cfg
        self.kwargs = kwargs

    def load_vocab(self):
        fn = os.path.join(
            self.config["name_or_path"], self.config.get("vocab_file", "vocab")
        )
        return load_vocab(fn)

    def get_vocab_size(self):
        return max(self.vocab_map.values()) + 1

    def get_all_vocab_ids(self):
        if self.all_token_ids is None:
            self.all_token_ids = tuple(range(self.get_vocab_size()))
        return self.all_token_ids

    def build_vocab(self):
        pass

    def get_label_pad_token(self):
        return "<label_pad>"

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

    def get_node_idx_tokens(self):
        if self.node_idx_tokens is None:
            self.node_idx_tokens = {
                str(x) for x in range(self.config["structure"]["node"]["scope_base"])
            }
        return self.node_idx_tokens

    def get_node_idx_token_ids(self):
        if self.node_idx_token_ids is None:
            self.node_idx_token_ids = {
                self.vocab_map[str(x)]
                for x in range(self.config["structure"]["node"]["scope_base"])
            }
        return self.node_idx_token_ids

    def get_token_components(self, ls_tokens):
        if self.token_components is None:
            one_token = ls_tokens[0]
            if isinstance(one_token, List):
                self.token_components = len(one_token)
            else:
                self.token_components = 0
        return self.token_components

    def get_label_token_id_to_be_padded(self):
        if self.task_type != "pretrain":
            label_token_ids = set([])
        else:
            label_tokens_to_be_padded = set(self.config.get("label_tokens_to_pad", []))
            label_token_ids = [
                self.vocab_map[token] for token in label_tokens_to_be_padded
            ]
            label_token_ids = set(label_token_ids)
        print(
            f"label token id to be converted to {self.label_pad_token_id} is {label_token_ids}"
        )
        return label_token_ids

    def _map_tokens_to_ids(self, tokens: Union[str, Iterable[str]]):
        if tokens is None:
            token_ids = None
        elif isinstance(tokens, str):
            token_ids = self.vocab_map[tokens]
        elif isinstance(tokens, Iterable):
            token_ids = tuple([self.vocab_map[token] for token in tokens])
        else:
            raise NotImplementedError(
                f"Not implement for type {type(tokens)} of tokens {tokens}!"
            )
        return token_ids

    def encode(self, seq):
        # input: tokenized sequence
        # output: id in vocabulary
        pass

    def pad(
        self,
        features: List[Dict],
        padding: bool = True,
        max_length: int = 128,
        pad_to_multiple_of: int = 8,
        return_tensors: str = "pt",
        mask_boundary: bool = False,
    ):
        # features:: list of input dicts
        # params setting is compatible with HF transformers
        assert return_tensors in {"pt", "np"}
        func = {"pt": torch.tensor, "np": np.array}[return_tensors]
        ls_seq_len = [len(feat["input_ids"]) for feat in features]
        pad_to = _get_batch_seq_len(ls_seq_len, pad_to_multiple_of, max_length)
        features = [self._pad_each_datapoint(feat, pad_to) for feat in features]

        batch_outputs = {}
        for feat in features:
            for key, value in feat.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        # sample_lens, split_lens and attn_modes stay as Python lists-of-lists (not tensors)
        _list_only_keys = {"sample_lens", "split_lens", "attn_modes"}
        batch_outputs = {
            key: (
                val if key in _list_only_keys
                else func(val) if not isinstance(val[0], str)
                else np.array(val)
            )
            for key, val in batch_outputs.items()
        }
        return batch_outputs

    def _pad_each_datapoint(self, feature, pad_to):
        if pad_to > len(feature["input_ids"]):
            padding_len = pad_to - len(feature["input_ids"])

            if isinstance(feature["input_ids"][0], Iterable):
                input_pad_val = [self.pad_token_id] * len(feature["input_ids"][0])
            else:
                assert isinstance(feature["input_ids"][0], int)
                input_pad_val = self.pad_token_id
            if isinstance(feature["labels"][0], Iterable):
                label_pad_val = [self.label_pad_token_id] * len(feature["labels"][0])
            else:
                assert isinstance(feature["labels"][0], int)
                label_pad_val = self.label_pad_token_id
            padded_input_ids = [input_pad_val] * padding_len
            padded_labels = [label_pad_val] * padding_len

            padded_nodev2_labels = [self.label_pad_token_id] * padding_len
            padded_position_ids = [0] * padding_len
            padded_attention_mask = [0] * padding_len

            feature["input_ids"] = _merge_two_ls(
                feature["input_ids"], padded_input_ids, self.padding_side
            )
            if "position_ids" in feature:
                feature["position_ids"] = _merge_two_ls(
                    feature["position_ids"], padded_position_ids, self.padding_side
                )
            feature["labels"] = _merge_two_ls(
                feature["labels"], padded_labels, self.padding_side
            )
            assert isinstance(
                feature["attention_mask"], List
            ), f"attention_mask type: {type(feature['attention_mask'])}"
            feature["attention_mask"] = _merge_two_ls(
                feature["attention_mask"], padded_attention_mask, self.padding_side
            )
            if "raw_node_idx" in feature:
                feature["raw_node_idx"] = _merge_two_ls(
                    feature["raw_node_idx"], padded_nodev2_labels, self.padding_side
                )
            set_vectors = {"embed", "noise"}
            for name in set_vectors:
                if name in feature:
                    dim = len(feature[name][0])
                    padded_vecs = np.zeros(
                        (padding_len, dim), dtype=np.float32
                    ).tolist()
                    feature[name] = _merge_two_ls(
                        feature[name], padded_vecs, self.padding_side
                    )
        else:
            keys_set = {
                "input_ids",
                "position_ids",
                "labels",
                "nodev2_labels",
                "raw_node_idx",
                "attention_mask",
                "embed",
                "noise",
            }
            for key, val in feature.items():
                if key in keys_set:
                    if isinstance(val, np.ndarray):
                        # For 2D block-wise attention mask in pre-train with packed sequence
                        assert key == "attention_mask", f"NOT for {key}"
                        feature[key] = val[:pad_to, :pad_to].tolist()
                    else:
                        feature[key] = val[:pad_to]
        return feature

    def pack_token_seq(
        self, token_res: TokenizationOutput, previous_idx: int
    ):
        ls_tokens = list(token_res.ls_tokens)
        ls_labels = list(token_res.ls_labels)
        ls_embed = list(token_res.ls_embed) if token_res.ls_embed else token_res.ls_embed
        token_components = self.get_token_components(ls_tokens)
        token_len = len(ls_tokens) + 1
        ls_len = [token_len]
        if isinstance(self.dataset, IterableDataset):
            print(f"Convert Iterable dataset to iterable: `dataset -> iter(dataset)`")
            self.dataset = iter(self.dataset)
        while token_len < self.mpe:
            if isinstance(self.dataset, Dataset):
                if random.uniform(0, 1.0) <= self.random_ratio:
                    # randomly sample a graph
                    idx = (
                        self.dataset.get_random_sample_idx()
                        if hasattr(self.dataset, "get_random_sample_idx")
                        else random.choice(self.sampler)
                    )
                else:  # repeat the previous graph
                    idx = previous_idx
                # sep_token = (
                #     self.get_eos_token()
                #     if idx != previous_idx
                #     else self.get_gsum_token()
                # )  # causing problem when pretrain-mlm
                sep_token = self.get_eos_token()
                _, new_graph = self.dataset[idx]
            else:
                idx = 0
                sep_token = self.get_eos_token()
                _, new_graph = next(self.dataset)
            token_res = self.tokenize(new_graph)
            new_ls_tokens = token_res.ls_tokens
            new_ls_labels = token_res.ls_labels
            new_ls_embed = token_res.ls_embed
            if token_components == 0:
                seps = [sep_token]
                label_seps = [self.get_label_pad_token()]
            else:
                seps = [[sep_token] * token_components]
                label_seps = [[self.get_label_pad_token()] * token_components]
            embed_seps = []
            if ls_embed:
                dim = len(ls_embed[0])
                embed_seps = np.zeros((1, dim), dtype=np.float32).tolist()
            # Drop this sample if it would exceed mpe
            if token_len + len(seps) + len(new_ls_tokens) >= self.mpe:
                break
            ls_tokens.extend(seps)
            ls_tokens.extend(new_ls_tokens)
            ls_labels.extend(label_seps)
            ls_labels.extend(new_ls_labels)
            if ls_embed:
                ls_embed.extend(embed_seps)
                ls_embed.extend(new_ls_embed)

            previous_idx = idx
            token_len = len(ls_tokens) + 1
            ls_len.append(token_len)
        return ls_tokens, ls_labels, ls_embed, ls_len

    def _tailor_node_struct_repr(self, node_structure_mapping):
        # for forward compatibility
        return node_structure_mapping

    def _tailor_edge_struct_repr(self, edge_structure_mapping):
        # for forward compatibility
        return edge_structure_mapping

    def tokenize(self, graph: Data):
        return self.raw_tokenize(graph)

    def raw_tokenize(self, graph: Data):
        # input: raw small/medium graph OR subgraph sampled from big graphs
        # output: sequence of tokens from vocab
        # 1~2. transform graph to Eulerian sequence
        assert (
            graph.num_nodes <= self.config["structure"]["node"]["node_scope"]
        ), f"num_nodes: {graph.num_nodes} > node_scope: {self.config['structure']['node']['node_scope']}"
        path = graph2path(graph)
        # 3. obtain node/edge structure and semantics mapping
        node_structure_mapping = nx_utils.get_structure_raw_node2idx_mapping(
            path,
            self.config["structure"]["node"]["scope_base"],
            self.config["structure"]["node"]["node_scope"],
            self.config["structure"]["node"].get("cyclic", False),
        )
        node_structure_mapping = self._tailor_node_struct_repr(node_structure_mapping)
        edge_structure_mapping = nx_utils.get_structure_raw_edge2type_mapping(
            path, graph
        )
        edge_structure_mapping = self._tailor_edge_struct_repr(edge_structure_mapping)
        (
            node_semantics_mapping,
            edge_semantics_mapping,
            graph_semantics_mapping,
        ) = self.semantics2tokens_mapping(path, graph, self.config)
        # 3.1 obtain target node or target edge tokens FOR node/edge-lvl tasks
        tgt_node_token = None
        tgt_edge_src_token = None
        tgt_edge_dst_token = None
        if hasattr(graph, "root_n_id"):
            # use re-indexed node-id to repr the node, e.g., 0/1/2/3/...
            if isinstance(graph.root_n_id, int):
                tgt_node_token = node_structure_mapping[graph.root_n_id]
            elif (
                isinstance(graph.root_n_id, torch.Tensor) and len(graph.root_n_id) == 2
            ):
                src, dst = graph.root_n_id.tolist()
                tgt_edge_src_token = node_structure_mapping[src]
                tgt_edge_dst_token = node_structure_mapping[dst]
            else:
                raise ValueError(
                    f"graph.root_n_id {graph.root_n_id} is not supported, Please check!"
                )
        # 4. decorate node/edge/graph with above mapping
        raw_seq = nx_utils.get_raw_seq_from_path(path)
        mask = get_mask_of_raw_seq(raw_seq, self.mask_type)
        (
            ls_tokens,
            ls_node_regression_labels,
            ls_edge_regression_labels,
        ) = nx_utils.decorate_node_edge_graph_with_mask(
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
        # 4.5 DEPRECATED@2024-10-22:: special decoration for molecule data with 3d coordinates
        # 5. remove bidirectional edge-type token, because it is treated as default edge-type,
        # keeping it will produce lots of redundant tokens
        dict_edge = self.config["structure"]["edge"]
        if dict_edge.get("remove_edge_type_token", False):
            edge_types = {self._tailor_edge_struct_repr(dict_edge["bi_token"])}
            ls_tokens = [token for token in ls_tokens if token not in edge_types]
        # 5.1 obtain label tokens from input tokens
        ls_labels = nx_utils.get_labels_from_input_tokens(ls_tokens, self)
        # 6. add nx/instructions/eos tokens and etc.
        # 6.1 enable nx func to enhance structure understanding
        ls_struct_tokens, ls_struct_labels = nx_utils.understand_structure(
            graph,
            tokenization_config=self.config,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            gtokenizer=self,
        )
        ls_tokens.extend(ls_struct_tokens)
        ls_labels.extend(ls_struct_labels)
        # 6.2 enable instruction tuning to enhance semantics understanding
        (
            ls_instruct_tokens,
            ls_instruct_labels,
        ) = instruct_tuning_utils.follow_instructions(
            graph,
            tokenization_config=self.config,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            gtokenizer=self,
        )
        ls_tokens.extend(ls_instruct_tokens)
        ls_labels.extend(ls_instruct_labels)
        # 6.3 add special tokens, e.g., eos
        if self.add_eos and ("pretrain" not in self.task_type):
            ls_tokens = self.add_eos_token(ls_tokens)
            ls_labels = self.add_eos_token(ls_labels)
        return TokenizationOutput(
            ls_tokens=ls_tokens,
            ls_labels=ls_labels,
            tgt_node_token=tgt_node_token,
            tgt_edge_src_token=tgt_edge_src_token,
            tgt_edge_dst_token=tgt_edge_dst_token,
            tgt_pos=None,  # TODO: implement ls_embed for GSTTokenizer
        )

    def convert_tokens_to_ids(self, seq_tokens, seq_labels):
        # 7. map tokens to token-id
        seq_tokens_id = [self.vocab_map[token] for token in seq_tokens]
        seq_labels_id = [self.vocab_map[token] for token in seq_labels]
        # 8. add labels, attention mask, position_ids and etc
        in_dict = get_input_dict_from_seq_tokens_id(
            seq_tokens_id,
            seq_labels_id,
            self.label_to_be_padded,
            self.label_pad_token_id,
        )
        return in_dict

    def prepare_inputs_for_task(
        self,
        in_dict: Dict,
        graph: Data,
        token_res: TokenizationOutput,
        is_training: Optional[bool] = None,
    ):
        tgt_node_token_id = self._map_tokens_to_ids(token_res.tgt_node_token)
        tgt_edge_src_token_id = self._map_tokens_to_ids(token_res.tgt_edge_src_token)
        tgt_edge_dst_token_id = self._map_tokens_to_ids(token_res.tgt_edge_dst_token)
        in_dict = prepare_inputs_for_task(
            self.task_type,
            in_dict,
            graph=graph,
            eos_token_id=self.get_eos_token_id(),
            tgt_node_token_id=tgt_node_token_id,
            tgt_edge_src_token_id=tgt_edge_src_token_id,
            tgt_edge_dst_token_id=tgt_edge_dst_token_id,
            tgt_pos=token_res.tgt_pos,
            gsum_token_id=self.get_gsum_token_id(),
            gtokenizer=self,
            ls_len=token_res.ls_len,
            ls_raw_node_idx=token_res.ls_raw_node_idx,
        )
        return in_dict

    def __call__(self, graph: Data, is_training: Optional[bool] = None):
        # 1~6. self.tokenize
        token_res = self.tokenize(graph)
        ls_tokens, ls_labels, ls_embed, ls_len = (
            self.pack_token_seq(token_res, graph.idx)
            if self.mpe is not None
            else (
                token_res.ls_tokens,
                token_res.ls_labels,
                token_res.ls_embed,
                [len(token_res.ls_tokens)],
            )
        )
        # 7~8. self.convert_tokens_to_ids
        in_dict = self.convert_tokens_to_ids(ls_tokens, ls_labels)
        if ls_embed:
            in_dict["embed"] = ls_embed
        # 9. prepare for tasks
        token_res.ls_tokens = ls_tokens
        token_res.ls_labels = ls_labels
        token_res.ls_embed = ls_embed
        token_res.ls_len = ls_len
        in_dict = self.prepare_inputs_for_task(
            in_dict,
            graph,
            token_res=token_res,
        )
        return in_dict

    def add_eos_token(self, seq_tokens):
        eos_token = self.get_eos_token()
        seq_tokens.append(eos_token)
        return seq_tokens

    def save_pretrained(self):
        pass


class StackedGSTTokenizer(GSTTokenizer):
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
        super().__init__(
            config,
            padding_side=padding_side,
            add_eos=add_eos,
            train_cfg=train_cfg,
            **kwargs,
        )
        assert stack_method in {"short", "long"}
        self.stack_method = stack_method
        self.default_node_attr = None
        self.default_edge_attr = None
        self.default_node_embed = None
        self.default_edge_embed = None
        self.default_edge_attr_id = None
        self.config["semantics"]["node"]["ignored_val"] = None
        self.config["semantics"]["edge"]["ignored_val"] = None
        assert rotation in DICT_pos_func.keys(), f"your rotation: {rotation}"
        self.rotation = rotation
        print(
            f"[StackedGSTTokenizer] stack_method: {stack_method}, rotation: {rotation}"
        )

    def get_default_node_attr(self, graph: Optional[Data] = None):
        if self.default_node_attr is None:
            self.default_node_attr = get_default_semantics_attr_mapping(
                graph, self.config, "node"
            )
        return self.default_node_attr

    def get_default_edge_attr(self, graph: Optional[Data] = None):
        if self.default_edge_attr is None:
            self.default_edge_attr = get_default_semantics_attr_mapping(
                graph, self.config, "edge"
            )
        return self.default_edge_attr

    def get_default_edge_attr_id(self, graph: Optional[Data] = None):
        if self.default_edge_attr_id is None:
            default_edge_attr = self.get_default_edge_attr(graph)
            self.default_edge_attr_id = self._map_tokens_to_ids(default_edge_attr)
        return self.default_edge_attr_id

    def get_default_node_embed(self, graph: Optional[Data] = None):
        if self.default_node_embed is None:
            self.default_node_embed = get_default_semantics_embed_mapping(
                graph, self.config, "node"
            )
        return self.default_node_embed

    def get_default_edge_embed(self, graph: Optional[Data] = None):
        if self.default_edge_embed is None:
            self.default_edge_embed = get_default_semantics_embed_mapping(
                graph, self.config, "edge"
            )
        return self.default_edge_embed

    def add_eos_token(self, ls_tokens):
        eos_token = self.config["structure"]["node"]["eos_token"]
        # ls = [eos_token] + self.get_default_node_attr() + self.get_default_edge_attr()
        ls = [eos_token] * len(ls_tokens[0])
        ls_tokens.append(ls)
        return ls_tokens

    def get_tokens_from_single_edge_attr(self, edge_attr: torch.Tensor):
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

    def tokenize(self, graph: Data):
        # input: raw small/medium graph OR subgraph sampled from big graphs
        # output: sequence of tokens from vocab
        if hasattr(graph, "pos") and graph.pos is not None:
            graph.pos = DICT_pos_func[self.rotation](graph.pos)
        if hasattr(graph, "rdkit_pos"):
            graph.rdkit_pos = DICT_pos_func[self.rotation](graph.rdkit_pos)
            graph.pos = torch.hstack([graph.pos, graph.rdkit_pos])
        # 1 & 2. get eulerian path
        path = graph2path(graph, prioritize=self.task_type != "pretrain")
        # 3. obtain node/edge structure and semantics mapping
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
        # below to be compatible with `instruct_tuning_utils._get_all_node_feats`
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
        # 3.1 obtain target node or target edge tokens FOR node/edge-lvl tasks
        tgt_node_token = None
        tgt_edge_src_token = None
        tgt_edge_dst_token = None
        tgt_edge_attr_token = None
        if hasattr(graph, "root_n_id"):
            # use re-indexed node-id to repr the node, e.g., 0/1/2/3/...
            if isinstance(graph.root_n_id, int):
                tgt_node_token = node_structure_mapping[graph.root_n_id]
            elif (
                isinstance(graph.root_n_id, torch.Tensor) and len(graph.root_n_id) == 2
            ):
                src, dst = graph.root_n_id.tolist()
                tgt_edge_src_token = node_structure_mapping[src]
                tgt_edge_dst_token = node_structure_mapping[dst]
                if hasattr(graph, "tgt_edge_attr"):
                    tgt_edge_attr_token = self.get_tokens_from_single_edge_attr(
                        graph.tgt_edge_attr
                    )
            else:
                raise ValueError(
                    f"graph.root_n_id {graph.root_n_id} is not supported, Please check!"
                )
        if self.task_type == "nodev2":
            assert tgt_node_token is None
            tgt_node_token = nx_utils._flatten_list(
                [node_structure_mapping[ele] for ele in range(graph.num_nodes)]
            )
        # 4. remove bidirectional edge-type token, because it is treated as default edge-type,
        # keeping it will produce redundant tokens
        if self.config["structure"]["edge"]["remove_edge_type_token"]:
            edge_structure_mapping = None
        # 5. stack node/edge/graph attr to nodes, so that total seq is Eulerian path len
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
        # 6. add special tokens, e.g., eos
        ls_tokens = self.add_eos_token(ls_tokens) if self.add_eos else ls_tokens
        ls_embed = add_eos_embed(ls_embed) if self.add_eos else ls_embed
        ls_raw_node_idx.append(-1) if self.add_eos else None
        token_components = self.get_token_components(ls_tokens)
        ls_labels = ls_tokens[1:] + [[self.get_eos_token()] * token_components]
        # 6.1 enable nx func to enhance structure understanding
        # TODO: implement it
        # 6.2 enable instruction tuning to enhance semantics understanding
        (
            ls_instruct_tokens,
            ls_instruct_labels,
        ) = instruct_tuning_utils.follow_instructions(
            graph,
            tokenization_config=self.config,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            gtokenizer=self,
        )
        ls_tokens.extend(ls_instruct_tokens) if len(ls_instruct_tokens) > 0 else None
        ls_labels.extend(ls_instruct_labels) if len(ls_instruct_labels) > 0 else None
        if ls_embed:
            assert (
                len(ls_instruct_tokens) == 0
            ), "NOT implemented when embed inputs is presented"
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

    def convert_tokens_to_ids(self, seq_tokens: List[List[str]], seq_labels: List[str]):
        # 7. map tokens to token-id
        seq_tokens_id = [
            [self.vocab_map[token] for token in feat_tokens]
            for feat_tokens in seq_tokens
        ]
        seq_labels_id = [
            [self.vocab_map[token] for token in feat_tokens]
            for feat_tokens in seq_labels
        ]
        # 8. add labels, attention mask, position_ids and etc
        in_dict = get_input_dict_from_seq_tokens_id(
            seq_tokens_id,
            seq_labels_id,
            set(),
            None,
        )
        # in_dict["labels"] = [token_ids[0] for token_ids in in_dict["labels"]]
        return in_dict

    def prepare_inputs_for_task(
        self,
        in_dict: Dict,
        graph: Data,
        token_res: TokenizationOutput,
        is_training: Optional[bool] = None,
    ):
        tgt_node_token_id = self._map_tokens_to_ids(token_res.tgt_node_token)
        tgt_edge_src_token_id = self._map_tokens_to_ids(token_res.tgt_edge_src_token)
        tgt_edge_dst_token_id = self._map_tokens_to_ids(token_res.tgt_edge_dst_token)
        tgt_edge_attr_token_id = self._map_tokens_to_ids(token_res.tgt_edge_attr_token)
        if self.stack_method == "long":
            if self.add_eos:
                ids = in_dict["input_ids"][:-1]
                eos = in_dict["input_ids"][-1:]
            else:
                ids = in_dict["input_ids"]
                eos = []
            node_attr_dim = self.config["semantics"]["node"]["dim"]
            in_dict["input_ids"] = [
                _pad_stacked_targets(
                    i,
                    ls_token_ids,
                    node_attr_dim=node_attr_dim,
                    padding_val=self.pad_token_id,
                    eos_token_id=self.get_eos_token_id(),
                )
                for i, ls_token_ids in enumerate(ids)
            ] + eos
        in_dict = prepare_inputs_for_task(
            self.task_type,
            in_dict,
            graph=graph,
            eos_token_id=self.get_eos_token_id(),
            tgt_node_token_id=tgt_node_token_id,
            tgt_edge_src_token_id=tgt_edge_src_token_id,
            tgt_edge_dst_token_id=tgt_edge_dst_token_id,
            tgt_edge_attr_token_id=tgt_edge_attr_token_id,
            tgt_pos=token_res.tgt_pos,
            gsum_token_id=self.get_gsum_token_id(),
            gtokenizer=self,
            ls_len=token_res.ls_len,
            ls_raw_node_idx=token_res.ls_raw_node_idx,
            **self.kwargs,
        )
        return in_dict
