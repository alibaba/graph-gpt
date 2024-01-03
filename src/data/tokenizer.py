import os
import sys
import random
import math
import numpy as np
from typing import Dict, List, Callable, Tuple, Optional, Union, Iterable, Set

import torch
from torch_geometric.data import Data
from ..utils import (
    mol_utils,
    get_edge_index,
    get_edge_type,
    graph2path,
    graph2path_test,
    get_precalculated_path,
    prepare_inputs_for_task,
    TASK_TYPES,
)
from .vocab_builder import load_vocab

# Below for SP tokenizer
DICT_col2repr = {
    "node": {i: chr(65 + i) for i in range(26)},
    "edge": {i: chr(97 + i) for i in range(26)},
    "graph": {0: "="},
}

DICT_edge_type2repr = {
    "<edge_jump>": ":",
    "<edge_in>": "<",
    "<edge_out>": ">",
    "<edge_bi>": "&",
}

NODE_repr = "#"
# above for SP tokenizer


class GSTTokenizer(object):
    def __init__(
        self, config: Dict, *, padding_side: str = "right", add_eos: bool = True
    ):
        self.config = config
        self.mask_type = self.config["semantics"].get("attr_assignment", "random")
        assert padding_side in {"left", "right"}
        self.padding_side = padding_side
        self.add_eos = add_eos
        self.vocab_map = self.load_vocab()
        self.vocab_size = max(self.vocab_map.values()) + 1
        self.label_pad_token_id = -100
        self.pad_token_id = 0
        self.task_type = self.config["task_type"].lower()
        assert self.task_type in TASK_TYPES, f"{self.task_type} is not implemented!"
        self.eos_idx = None
        # below for pack target token sequence with randomly sampled token sequence
        self.mpe = None
        self.dataset = None
        self.sampler = None
        self.random_ratio = 1
        self.label_to_be_padded = self.get_label_token_id_to_be_padded()

    def load_vocab(self):
        fn = os.path.join(
            self.config["name_or_path"], self.config.get("vocab_file", "vocab")
        )
        return load_vocab(fn)

    def build_vocab(self):
        pass

    def get_bos_token(self):
        return self.config["structure"]["node"]["bos_token"]

    def get_eos_token(self):
        return self.config["structure"]["node"]["eos_token"]

    def get_gsum_token(self):
        return self.config["structure"]["graph"]["summary_token"]

    def get_icl_token(self):
        return self.config["structure"]["common"]["icl_token"]

    def get_sep_token(self):
        return self.config["structure"]["common"]["sep_token"]

    def get_common_semantics(self):
        return self.config["semantics"]["common"].get("reserved_token", [])

    def get_bos_token_id(self):
        return self.vocab_map[self.get_bos_token()]

    def get_eos_token_id(self):
        return self.vocab_map[self.get_eos_token()]

    def get_gsum_token_id(self):
        return self.vocab_map.get(self.get_gsum_token(), None)

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

    def set_eos_idx(self, input_ids: List[int]):
        if self.eos_idx is None:
            # every worker of Loader will calculate eos_idx independently!
            eos_token_id = self.get_eos_token_id()
            assert isinstance(input_ids, list)
            try:
                if self.mpe is None:
                    # NO packing of token sequence
                    idx = input_ids.index(eos_token_id)
                    self.eos_idx = idx - len(input_ids)
                else:
                    # PACKING token sequences with <eos_token> as sep
                    self.eos_idx = int(1e8)
            except ValueError as e:
                # print(f"ValueError: {e}:: {input_ids}")
                self.eos_idx = int(1e8)
            finally:
                print(
                    f"[Warning] Set eos_idx to {self.eos_idx} for task {self.task_type}!"
                )
            # print(f"SET eos_idx to be {self.eos_idx}")
            # This will be executed every epoch at every worker of the loader

    def pad(
        self,
        features: List[Dict],
        padding: bool = True,
        max_length: int = 128,
        pad_to_multiple_of: int = 8,
        return_tensors: str = "pt",
    ):
        # features:: list of input dicts
        # params setting is compatible with HF transformers
        self.set_eos_idx(features[0]["input_ids"])
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
        batch_outputs = {key: func(val) for key, val in batch_outputs.items()}
        if "attention_mask_bi" in batch_outputs:
            attention_mask = batch_outputs["attention_mask"]
            attention_mask_bi = batch_outputs["attention_mask_bi"]
            assert (
                attention_mask | attention_mask_bi
            ).sum().item() == attention_mask.sum().item()
        return batch_outputs

    def _pad_each_datapoint(self, feature, pad_to):
        if pad_to > len(feature["input_ids"]):
            padding_len = pad_to - len(feature["input_ids"])

            padded_input_ids = [self.pad_token_id] * padding_len
            padded_position_ids = [0] * padding_len
            padded_labels = [self.label_pad_token_id] * padding_len
            padded_attention_mask = [0] * padding_len

            feature["input_ids"] = _merge_two_ls(
                feature["input_ids"], padded_input_ids, self.padding_side
            )
            feature["position_ids"] = _merge_two_ls(
                feature["position_ids"], padded_position_ids, self.padding_side
            )
            feature["labels"] = _merge_two_ls(
                feature["labels"], padded_labels, self.padding_side
            )
            feature["attention_mask"] = _merge_two_ls(
                feature["attention_mask"], padded_attention_mask, self.padding_side
            )
            if "attention_mask_bi" in feature:
                feature["attention_mask_bi"] = _merge_two_ls(
                    feature["attention_mask_bi"],
                    padded_attention_mask,
                    self.padding_side,
                )
            if "pos" in feature:
                padded_pos = torch.zeros(padding_len, 3, dtype=torch.float32)
                tensors = (
                    [feature["pos"], padded_pos]
                    if self.padding_side == "right"
                    else [padded_pos, feature["pos"]]
                )
                feature["pos"] = torch.cat(tensors, dim=0).tolist()
        else:
            keys_set = {
                "input_ids",
                "position_ids",
                "labels",
                "attention_mask",
                "attention_mask_bi",
            }
            # eos_idx<0 or eos_idx=1e8
            # the design of negative `eos_idx` is to keep the task-specific tails
            # e.g., tail `<eos> <tgt-node>` for node task -> eos_idx=-2
            # tail `<eos> <src-node> <dst-node>` for edge task -> eos_idx=-3
            # tail `<eos> <gsum>` for graph task -> eos_idex=-2
            mid_idx = pad_to + self.eos_idx if self.eos_idx < 0 else pad_to
            for key, val in feature.items():
                feature[key] = (
                    val[:mid_idx] + val[self.eos_idx :] if key in keys_set else val
                )
            if "pos" in feature:
                feature["pos"] = feature["pos"][:pad_to].tolist()
        return feature

    def pack_token_seq(self, ls_tokens: List[str], previous_idx: int):
        token_len = len(ls_tokens) + 1
        while token_len < self.mpe:
            if random.uniform(0, 1.0) <= self.random_ratio:
                # randomly sample a graph
                idx = (
                    self.dataset.get_random_sample_idx()
                    if hasattr(self.dataset, "get_random_sample_idx")
                    else random.choice(self.sampler)
                )
            else:  # repeat the previous graph
                idx = previous_idx
            sep_token = (
                self.get_eos_token() if idx != previous_idx else self.get_gsum_token()
            )
            _, new_graph = self.dataset[idx]
            new_ls_tokens, _, _, _, _ = self.tokenize(new_graph)
            ls_tokens = ls_tokens + [sep_token] + new_ls_tokens

            previous_idx = idx
            token_len = len(ls_tokens) + 1
        return ls_tokens

    def tokenize(self, graph: Data):
        # input: raw small/medium graph OR subgraph sampled from big graphs
        # output: sequence of tokens from vocab
        if "paths_ind" in graph:
            # 1 & 2. Retrieve a path from pre-calculcated paths
            path = get_precalculated_path(graph)
        else:
            path = graph2path(graph, prioritize=self.task_type != "pretrain")
        # 3. obtain node/edge structure and semantics mapping
        node_structure_mapping = get_structure_raw_node2idx_mapping(
            path, self.config["structure"]["node"]["scope_base"]
        )
        edge_structure_mapping = get_structure_raw_edge2type_mapping(path, graph)
        (
            node_semantics_mapping,
            edge_semantics_mapping,
            graph_semantics_mapping,
        ) = get_semantics_raw_node_edge2attr_mapping(path, graph, self.config)
        # 3.1 obtain target node or target edge tokens FOR node/edge-lvl tasks
        tgt_node_token = None
        tgt_edge_src_token = None
        tgt_edge_dst_token = None
        if hasattr(graph, "root_n_id"):
            # use local node-id to repr the node, e.g., 1/2/3/...
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
        # tgt_node_token = (
        #     node_semantics_mapping["discrete"][graph.root_n_id][-1]
        #     if hasattr(graph, "root_n_id")
        #     else None
        # )  # use this global node-id to repr the node for one-big-graph :: NOT helpful for node-level task
        # 4. decorate node/edge/graph with above mapping
        raw_seq = get_raw_seq_from_path(path)
        mask = get_mask_of_raw_seq(raw_seq, self.mask_type)
        (
            ls_tokens,
            ls_node_regression_labels,
            ls_edge_regression_labels,
        ) = decorate_node_edge_graph_with_mask(
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
        # 4.5 special decoration for molecule data with 3d coordinates
        added_tokens = (
            decorate_molecules_with_3d_positions(graph, self, node_structure_mapping)
            if self.config["semantics"].get("add_3d", False)
            else []
        )
        ls_tokens.extend(added_tokens)
        tgt_pos = None
        if self.config["semantics"].get("3d_regression", False):
            tgt_pos, _ = _trans_rot_pos(graph, node_structure_mapping)  # [num_nodes, 3]
        # 5. remove bidirectional edge-type token, because it is treated as default edge-type,
        # keeping it will produce lots of redundant tokens
        dict_edge = self.config["structure"]["edge"]
        if dict_edge.get("remove_edge_type_token", False):
            edge_types = {dict_edge["bi_token"]}
            ls_tokens = [token for token in ls_tokens if token not in edge_types]
        # 6. add special tokens, e.g., eos
        ls_tokens = self.add_eos_token(ls_tokens) if self.add_eos else ls_tokens
        return (
            ls_tokens,
            tgt_node_token,
            tgt_edge_src_token,
            tgt_edge_dst_token,
            tgt_pos,
        )

    def convert_tokens_to_ids(self, seq_tokens):
        # 7. map tokens to token-id
        seq_tokens_id = [self.vocab_map[token] for token in seq_tokens]
        # 8. add labels, attention mask, position_ids and etc
        in_dict = get_input_dict_from_seq_tokens_id(
            seq_tokens_id, self.label_to_be_padded, self.label_pad_token_id
        )
        return in_dict

    def prepare_inputs_for_task(
        self,
        in_dict: Dict,
        graph: Data,
        tgt_node_token: Union[str, List[str], Tuple[str]],
        tgt_edge_src_token: Union[str, List[str], Tuple[str]],
        tgt_edge_dst_token: Union[str, List[str], Tuple[str]],
        tgt_pos: Optional[torch.Tensor] = None,
    ):
        tgt_node_token_id = self._map_tokens_to_ids(tgt_node_token)
        tgt_edge_src_token_id = self._map_tokens_to_ids(tgt_edge_src_token)
        tgt_edge_dst_token_id = self._map_tokens_to_ids(tgt_edge_dst_token)
        in_dict = prepare_inputs_for_task(
            self.task_type,
            in_dict,
            graph=graph,
            eos_token_id=self.get_eos_token_id(),
            tgt_node_token_id=tgt_node_token_id,
            tgt_edge_src_token_id=tgt_edge_src_token_id,
            tgt_edge_dst_token_id=tgt_edge_dst_token_id,
            tgt_pos=tgt_pos,
            gsum_token_id=self.get_gsum_token_id(),
            gtokenizer=self,
        )
        return in_dict

    def __call__(self, graph: Data):
        # 1~6. self.tokenize
        (
            ls_tokens,
            tgt_node_token,
            tgt_edge_src_token,
            tgt_edge_dst_token,
            tgt_pos,
        ) = self.tokenize(graph)
        ls_tokens = (
            self.pack_token_seq(ls_tokens, graph.idx)
            if self.mpe is not None
            else ls_tokens
        )
        # 7~8. self.convert_tokens_to_ids
        in_dict = self.convert_tokens_to_ids(ls_tokens)
        # 9. prepare for tasks
        in_dict = self.prepare_inputs_for_task(
            in_dict,
            graph,
            tgt_node_token,
            tgt_edge_src_token,
            tgt_edge_dst_token,
            tgt_pos,
        )
        return in_dict

    def add_eos_token(self, seq_tokens):
        eos_token = self.config["structure"]["node"]["eos_token"]
        seq_tokens.append(eos_token)
        return seq_tokens

    def save_pretrained(self):
        pass


def _merge_two_ls(ls_main, ls_side, side="left"):
    return ls_side + ls_main if side == "left" else ls_main + ls_side


def _get_batch_seq_len(ls_seq_len, pad_to_multiple_of, max_position_embeddings):
    if pad_to_multiple_of is None:
        batch_seq_len = max_position_embeddings
    else:
        max_seq_len = max(ls_seq_len)
        batch_seq_len = pad_to_multiple_of * int(
            math.ceil(max_seq_len / pad_to_multiple_of)
        )
        batch_seq_len = min(batch_seq_len, max_position_embeddings)
    return batch_seq_len


def get_input_dict_from_seq_tokens_id(
    seq_tokens_id: List[int], label_to_be_pad: Set[int], label_pad_token_id: int
):
    seq_tokens_id = list(seq_tokens_id)
    input_ids = seq_tokens_id[:-1]
    labels = seq_tokens_id[1:]
    if len(label_to_be_pad) > 0:
        labels = [
            token_id if token_id not in label_to_be_pad else label_pad_token_id
            for token_id in labels
        ]
    position_ids = list(range(len(input_ids)))
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def _rebase_idx(idx: int, base: int):
    assert idx < base * base
    idx_1 = idx // base
    idx_2 = idx - idx_1 * base
    rebased_idx = (f"{idx_1}*{base}", str(idx_2)) if idx_1 > 0 else (str(idx_2),)
    return rebased_idx


def get_structure_raw_node2idx_mapping(path: List[Tuple[int]], scope_base: int):
    # refer: https://stackoverflow.com/a/17016257/4437068
    assert (sys.version_info.major == 3) and (sys.version_info.minor >= 7)
    if path:
        path_s = [src for src, tgt in path]
        path_s.append(path[-1][-1])
        uniques = list(dict.fromkeys(path_s))
        dict_map = {
            old_idx: _rebase_idx(idx, scope_base) for idx, old_idx in enumerate(uniques)
        }
        # 1st element starts from 0
    else:  # in case `path=[]` when graph has ONLY 1 node
        dict_map = {0: ("0",)}
    return dict_map


def get_structure_raw_edge2type_mapping(path: List[Tuple[int]], data: Data):
    # map the edge to its type
    dict_map = {
        (src, tgt): get_edge_type(data.edge_index, src, tgt) for src, tgt in path
    }
    return dict_map


def _tokenize_discrete_attr(
    raw_attr: List[str],
    world_identifier: str,
    node_edge_identifier: str,
    ignored_val: str = None,
    shuffle: bool = False,
):
    # input:: raw_attr: e.g., [4932, 29376]
    # output:: e.g., ['ogbn-proteins#node#0#4932', 'ogbn-proteins#node#1#29376']
    tokens = [
        f"{world_identifier}#{node_edge_identifier}#{col_idx}#{col_val}"
        for col_idx, col_val in enumerate(raw_attr)
        if col_val != str(ignored_val)
    ]
    if shuffle:
        random.shuffle(tokens)
    return tokens


def _remove_lead_zero(ls_col_val):
    # remove leading 0 to reduce token length if it is decimals < 1
    return (
        ls_col_val[1:]
        if (len(ls_col_val) > 2) and (ls_col_val[0] == "0") and (ls_col_val[1] == ".")
        else ls_col_val
    )


def _tokenize_continuous_attr(
    raw_attr: List[str],
    world_identifier: str,
    node_edge_identifier: str,
    ignored_val: str = None,
    shuffle: bool = False,
):
    # input:: raw_attr: e.g., ['500', '0', '380']
    # output:: e.g., ['ogbn-proteins#node#0#1', '<5>', '<0>', '<0>', 'ogbn-proteins#node#2#1', '<3>', '<8>', '<0>']
    def _process_each_col(col_idx, col_val):
        ls_col_val = list(col_val)
        ls_col_val = _remove_lead_zero(ls_col_val)
        ls_col_val = [f"<{x}>" for x in ls_col_val]
        identifier = (
            f"{world_identifier}#{node_edge_identifier}#{col_idx}#1"
            if node_edge_identifier != "graph"
            else "<gsum>"
        )
        return [identifier] + ls_col_val

    tokens = [
        _process_each_col(col_idx, col_val)
        for col_idx, col_val in enumerate(raw_attr)
        if col_val != str(ignored_val)
    ]
    if shuffle:
        random.shuffle(tokens)
    return tokens


def _add_regression_token(dict_map, reg_token):
    for val in dict_map.values():
        val.append(reg_token)


def _get_node2attr_mapping(path, data: Data, attr_name: str):
    if path:
        tmp_map = {}
        for src, tgt in path:
            if tmp_map.get(src, None) is None:
                tmp_map[src] = data[attr_name][src].numpy().astype(str)

        if (
            tmp_map.get(tgt, None) is None
        ):  # for semi-euler path OR shortened euler path, which does not go back to origin
            tmp_map[tgt] = data[attr_name][tgt].numpy().astype(str)
    else:  # in case `path=[]` when graph has ONLY 1 node
        node = 0
        tmp_map = {node: data[attr_name][node].numpy().astype(str)}
    return tmp_map


def _get_edge2attr_mapping(path, data: Data, attr_name: str, verbose: bool = False):
    tmp_map = {}
    for src, tgt in path:
        idx = get_edge_index(data.edge_index, src, tgt)
        if idx.shape[0] == 0:
            idx_backward = get_edge_index(data.edge_index, tgt, src)
            if idx_backward.shape[0] == 0:
                idx = None
                print(
                    f"Edge ({src}, {tgt}) or ({tgt}, {src}) does not have attr {attr_name}"
                ) if verbose else None
            else:
                idx = idx_backward
        if idx is not None:
            idx = idx.item()
            attr_val = data[attr_name][idx]
            tmp_map[(src, tgt)] = attr_val.numpy().astype(str)
    return tmp_map


def _get_graph2attr_mapping(path, data: Data, attr_name: str):
    return {0: data[attr_name][0].numpy().astype(str)}


def get_semantics_attr_mapping(
    path, data: Data, config: Dict, node_or_edge: str, func_attr_mapping: Callable
):
    # input: path
    # output: a mapping of each node/edge to its attr, and each node to its global-idx if exists
    assert node_or_edge in {"node", "edge", "graph"}
    dict_map = {"discrete": {}, "continuous": {}}
    attr_shuffle = config["semantics"].get("attr_shuffle", False)

    discrete_attr = config["semantics"][node_or_edge]["discrete"]
    world_identifier = config["attr_world_identifier"]
    if discrete_attr is not None:
        ignored_val = config["semantics"][node_or_edge]["ignored_val"]
        tmp_map = func_attr_mapping(path, data, discrete_attr)
        dict_map["discrete"] = {
            k: _tokenize_discrete_attr(
                v, world_identifier, node_or_edge, ignored_val, attr_shuffle
            )
            for k, v in tmp_map.items()
        }

    continuous_attr = config["semantics"][node_or_edge]["continuous"]
    if continuous_attr is not None:
        assert (
            discrete_attr is None
        ), "Supporting both discrete and continuous attr is NOT implemented yet!"
        ignored_val = config["semantics"][node_or_edge]["ignored_val"]
        tmp_map = func_attr_mapping(path, data, continuous_attr)
        dict_map["discrete"] = {
            k: _tokenize_continuous_attr(
                v, world_identifier, node_or_edge, ignored_val, attr_shuffle
            )
            for k, v in tmp_map.items()
        }
    return dict_map


def get_semantics_raw_node_edge2attr_mapping(path, data: Data, config: Dict):
    dict_map_node = get_semantics_attr_mapping(
        path, data, config, "node", _get_node2attr_mapping
    )
    dict_map_edge = get_semantics_attr_mapping(
        path, data, config, "edge", _get_edge2attr_mapping
    )
    dict_map_graph = get_semantics_attr_mapping(
        path, data, config, "graph", _get_graph2attr_mapping
    )
    return dict_map_node, dict_map_edge, dict_map_graph


def get_raw_seq_from_path(path):
    # raw_seq:: [<node>, <edge>, <node>, <edge>, ...]
    # <node> in the format of int, e.g., 3
    # <edge> in the format of tuple of int, e.g., (3, 0)
    raw_seq = []
    if path:
        for src, tgt in path:
            raw_seq.append(src)
            raw_seq.append((src, tgt))
        raw_seq.append(tgt)
    else:  # in case `path=[]` when graph has ONLY 1 node
        raw_seq.append(0)
    return raw_seq


def _obtain_all_idx_of_each_element(seq: List):
    dict_idx = {}
    for i, ele in enumerate(seq):
        if ele not in dict_idx:
            dict_idx[ele] = []
        dict_idx[ele].append(i)
    return dict_idx


def _obtain_first_appearance_idx(dict_idx):
    return [val[0] for val in dict_idx.values()]


def _obtain_last_appearance_idx(dict_idx):
    return [val[-1] for val in dict_idx.values()]


def _obtain_random_appearance_idx(dict_idx):
    return [random.choice(val) for val in dict_idx.values()]


def _obtain_all_appearance_idx(dict_idx):
    return [idx for val in dict_idx.values() for idx in val]


def get_mask_of_raw_seq(raw_seq, mask_type="first"):
    deco_seq = [
        (min(ele), max(ele)) if isinstance(ele, tuple) else ele for ele in raw_seq
    ]
    dict_deco_idx = _obtain_all_idx_of_each_element(deco_seq)
    dict_mask_func = {
        "first": _obtain_first_appearance_idx,
        "last": _obtain_last_appearance_idx,
        "random": _obtain_random_appearance_idx,
        "all": _obtain_all_appearance_idx,
    }
    mask_func = dict_mask_func[mask_type]
    idx = mask_func(dict_deco_idx)
    idx = sorted(idx)

    seq_len = len(raw_seq)
    mask = np.zeros(seq_len, dtype=int)
    mask[idx] = 1
    return mask


def _unfold_ls_of_ls(ls, shuffle: bool = False):
    random.shuffle(ls) if shuffle else None
    if isinstance(ls[0], list) or isinstance(ls[0], tuple):
        ls = [item for row in ls for item in row]
    return ls


def decorate_node_edge_graph_with_mask(
    gtokenizer: GSTTokenizer,
    raw_seq,
    mask,
    node_structure_mapping,
    edge_structure_mapping,
    node_semantics_mapping,
    edge_semantics_mapping,
    graph_semantics_mapping,
    attr_shuffle: bool = False,
):
    ls_tokens = []  # For next-token-prediction
    ls_node_regression_labels = []  # Groundtruth for predict continuous node attr
    ls_edge_regression_labels = []  # Groundtruth for predict continuous edge attr
    for i, (raw_token, is_deco) in enumerate(zip(raw_seq, mask)):
        if i % 2 == 0:  # deco node
            node_id = node_structure_mapping[raw_token]
            # node_id will be List if it is represented by several ids, e.g., global+local-ids
            ls_tokens.extend(node_id) if isinstance(
                node_id, Iterable
            ) else ls_tokens.append(node_id)
            if is_deco:
                node_attr = node_semantics_mapping["discrete"].get(raw_token, None)
                if node_attr:
                    ls_tokens.extend(_unfold_ls_of_ls(node_attr, attr_shuffle))
        else:  # deco edge
            edge_type = edge_structure_mapping[raw_token]
            ls_tokens.append(edge_type)
            if is_deco:
                edge_attr = edge_semantics_mapping["discrete"].get(raw_token, None)
                if edge_attr:
                    ls_tokens.extend(_unfold_ls_of_ls(edge_attr, attr_shuffle))
    # deco graph
    graph_attr = graph_semantics_mapping["discrete"].get(0, [])
    ls_tokens.extend(
        [gtokenizer.get_eos_token()] + _unfold_ls_of_ls(graph_attr, False)
    ) if len(graph_attr) > 0 else None
    return ls_tokens, ls_node_regression_labels, ls_edge_regression_labels


def _trans_rot_pos(
    graph,
    node_structure_mapping: Dict[int, Tuple[str]],
):
    # 1. get the coord of nodes in the order of euler sequence
    idx_euler_order = [
        old_idx
        for old_idx, new_idx in sorted(
            node_structure_mapping.items(), key=lambda x: eval("+".join(x[1]))
        )
    ]
    if (
        hasattr(graph, "pos")
        and isinstance(graph.pos, torch.Tensor)
        and (not torch.isnan(graph.pos).any().item())
    ):
        pos = graph.pos[idx_euler_order]

        # 2. reset the coord by subtracting 1st node's coord in euler sequence
        pos = pos - pos[0:1]  # [N, 3]  translational invariant
    else:
        pos = torch.tensor([0, 0, 0], dtype=torch.float32)

    if torch.abs(pos).sum() > 1e-8:
        pos = mol_utils.rotate_3d_v3(pos)  # [N,3]  rotational invariant
    return pos, idx_euler_order


def decorate_molecules_with_3d_positions(
    graph,
    gtokenizer: GSTTokenizer,
    node_structure_mapping: Dict[int, Tuple[str]],
    trim_zeros: bool = True,
):
    # 1. define two special tokens: icl_token, sep_token
    eos_token = gtokenizer.get_eos_token()
    icl_token = gtokenizer.get_icl_token()
    sep_token = gtokenizer.get_sep_token()

    # 2&3. translational & rotational invariant transformation of 3d-pos
    pos, idx_euler_order = _trans_rot_pos(graph, node_structure_mapping)

    # 4. obtain the tokens if the graph has 3d coordinate
    def _process_each_col(col_val: str):
        ls_col_val = list(col_val)
        # ls_col_val = _remove_lead_zero(ls_col_val)
        return [f"<{x}>" for x in ls_col_val]

    if torch.abs(pos).sum() > 1e-8:
        decimals = gtokenizer.config["semantics"].get("3d_decimals", 2)
        ls_coords = []
        pos_str = pos.round(decimals=decimals).numpy().astype(str)
        for idx, node_pos in enumerate(
            pos_str[1:], 1
        ):  # 1st node is translated to (0,0,0), so ignore it
            coords = [[sep_token] + _process_each_col(col_val) for col_val in node_pos]
            # 2nd node is rotated to (0,0,z), so ignore x,y coords
            # 3rd node is rotated to (0,y,z), so ignore x coords
            trim = max(0, 3 - idx) if trim_zeros else 0
            coords = coords[trim:]
            coords_flat = [token for each_coord in coords for token in each_coord]
            coords_flat = coords_flat[1:]  # remove the leading sep_token

            old_idx = idx_euler_order[idx]
            new_idx = node_structure_mapping[old_idx]

            node_with_coords = list(new_idx) + coords_flat
            ls_coords.append(node_with_coords)
            # print(node_with_coords)
        ls_coords = [token for each in ls_coords for token in each]
        ls_coords = [eos_token, icl_token] + ls_coords
    else:
        ls_coords = []
    return ls_coords


class SPGSTTokenizer(GSTTokenizer):
    def __init__(
        self, config: Dict, *, padding_side: str = "right", add_eos: bool = True
    ):
        super().__init__(config, padding_side=padding_side, add_eos=add_eos)

    def tokenize(self, graph: Data):
        # input: raw small/medium graph OR subgraph sampled from big graphs
        # output: sequence of tokens from vocab
        # 1 & 2. get Eulerian path from graph
        path = graph2path(graph, prioritize=self.task_type != "pretrain")
        # 3. obtain node/edge structure and semantics mapping
        # 3.1 structure mapping
        node_structure_mapping = get_structure_raw_node2idx_mapping(
            path, self.config["structure"]["node"]["scope_base"]
        )
        node_structure_mapping = {
            k: f"{NODE_repr}{v}" for k, v in node_structure_mapping.items()
        }
        edge_structure_mapping = get_structure_raw_edge2type_mapping(path, graph)
        edge_structure_mapping = {
            k: DICT_edge_type2repr[v] for k, v in edge_structure_mapping.items()
        }
        # 3.2 semantics mapping
        (
            sp_node_semantics_mapping,
            sp_edge_semantics_mapping,
            sp_graph_semantics_mapping,
        ) = get_sp_semantics_raw_node_edge_graph2attr_mapping(path, graph, self.config)
        # 4. decorate node/edge/graph with above mapping
        raw_seq = get_raw_seq_from_path(path)
        mask = get_mask_of_raw_seq(raw_seq, self.mask_type)
        (
            ls_tokens,
            ls_node_regression_labels,
            ls_edge_regression_labels,
        ) = decorate_node_edge_graph_with_mask(
            self,
            raw_seq,
            mask,
            node_structure_mapping,
            edge_structure_mapping,
            sp_node_semantics_mapping,
            sp_edge_semantics_mapping,
            sp_graph_semantics_mapping,
            attr_shuffle=self.config["semantics"].get("attr_shuffle", False),
        )
        # 5. remove bidirectional edge-type token, because it is treated as default edge-type,
        # keeping it will produce lots of redundant tokens
        dict_edge = self.config["structure"]["edge"]
        if dict_edge.get("remove_edge_type_token", False):
            edge_types = {DICT_edge_type2repr[dict_edge["bi_token"]]}
            ls_tokens = [token for token in ls_tokens if token not in edge_types]
        # 6. add special tokens, e.g., eos
        ls_tokens = self.add_eos_token(ls_tokens) if self.add_eos else ls_tokens
        return ls_tokens, None, None, None

    def add_eos_token(self, seq_tokens):
        eos_token = self.config["structure"]["node"]["eos_token"]
        seq_tokens.append(f"{NODE_repr}{eos_token}")
        return seq_tokens

    def load_vocab(self):
        fn = os.path.join(
            self.config["name_or_path"], self.config.get("vocab_file", "vocab")
        )
        return load_vocab(fn)


def _sp_tokenize_attr(
    raw_attr: List[str],
    node_edge_identifier: str,
    ignored_val: str = None,
    shuffle: bool = False,
):
    # input:: raw_attr: e.g., ['500', '0', '380']
    # output:: e.g., ['A500', 'C380']
    dict_col2repr = DICT_col2repr[node_edge_identifier]
    tokens = [
        dict_col2repr[col_idx] + col_val
        for col_idx, col_val in enumerate(raw_attr)
        if col_val != str(ignored_val)
    ]
    if shuffle:
        random.shuffle(tokens)
    return tokens


def get_sp_semantics_attr_mapping(
    path, data: Data, config: Dict, node_or_edge: str, func_attr_mapping: Callable
):
    # input: path
    # output: a mapping of each node/edge to its attr, and each node to its global-idx if exists
    assert node_or_edge in {"node", "edge", "graph"}

    attr_shuffle = config["semantics"].get("attr_shuffle", False)

    attr = config["semantics"][node_or_edge]["continuous"]
    if attr is not None:
        ignored_val = config["semantics"][node_or_edge]["ignored_val"]
        tmp_map = func_attr_mapping(path, data, attr)
        dict_map = {
            k: _sp_tokenize_attr(v, node_or_edge, ignored_val, attr_shuffle)
            for k, v in tmp_map.items()
        }
    else:
        # print(f"[Warning]attr {attr} is None") if attr is None else None
        dict_map = {}
    return {"discrete": dict_map, "continuous": {}}


def get_sp_semantics_raw_node_edge_graph2attr_mapping(path, data: Data, config: Dict):
    dict_map_node = get_sp_semantics_attr_mapping(
        path, data, config, "node", _get_node2attr_mapping
    )
    dict_map_edge = get_sp_semantics_attr_mapping(
        path, data, config, "edge", _get_edge2attr_mapping
    )
    dict_map_graph = get_sp_semantics_attr_mapping(
        path, data, config, "graph", _get_graph2attr_mapping
    )
    return dict_map_node, dict_map_edge, dict_map_graph
