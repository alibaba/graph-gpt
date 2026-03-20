"""Graph-to-token encoding: attribute tokenization and semantics mapping."""

import random
from typing import Callable, Dict, List

import numpy as np
from torch_geometric.data import Data

from ...utils import nx_utils


def _remove_lead_zero(ls_col_val):
    """Remove leading 0 to reduce token length if it is decimals < 1."""
    return (
        ls_col_val[1:]
        if (len(ls_col_val) > 2) and (ls_col_val[0] == "0") and (ls_col_val[1] == ".")
        else ls_col_val
    )


def _tokenize_discrete_attr(
    raw_attr: List[str],
    world_identifier: str,
    node_edge_identifier: str,
    ignored_val: str = None,
    shuffle: bool = False,
    remove_val: bool = False,
    share_vocab: bool = False,
):
    # input:: raw_attr: e.g., [4932, 29376]
    # output:: e.g., ['ogbn-proteins#node#0#4932', 'ogbn-proteins#node#1#29376']
    #            OR  ['ogbn-proteins#node#0', 'ogbn-proteins#node#1']
    if remove_val:
        tokens = [
            f"{world_identifier}#{node_edge_identifier}#-1"
            if share_vocab
            else f"{world_identifier}#{node_edge_identifier}#{col_idx}"
            for col_idx, _ in enumerate(raw_attr)
        ]
    else:
        tokens = [
            f"{world_identifier}#{node_edge_identifier}#-1#{col_val}"
            if share_vocab
            else f"{world_identifier}#{node_edge_identifier}#{col_idx}#{col_val}"
            for col_idx, col_val in enumerate(raw_attr)
            if col_val != str(ignored_val)
        ]
        if shuffle:
            random.shuffle(tokens)
    return tokens


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
                tmp_map[src] = data[attr_name][src].numpy()

        # for semi-euler path OR shortened euler path, which does not go back to origin
        if tmp_map.get(tgt, None) is None:
            tmp_map[tgt] = data[attr_name][tgt].numpy()
    else:  # in case `path=[]` when graph has ONLY 1 node
        node = 0
        tmp_map = {node: data[attr_name][node].numpy()}
    return tmp_map


def _get_edge2attr_mapping(path, data: Data, attr_name: str, verbose: bool = False):
    tmp_map = {}
    for src, tgt in path:
        idx = nx_utils.get_edge_index(data.edge_index, src, tgt)
        if idx.shape[0] == 0:
            idx_backward = nx_utils.get_edge_index(data.edge_index, tgt, src)
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
            tmp_map[(src, tgt)] = attr_val.numpy()
    return tmp_map


def _get_graph2attr_mapping(path, data: Data, attr_name: str):
    return {0: data[attr_name][0].numpy()}


def get_semantics_attr_mapping(
    path, data: Data, config: Dict, node_or_edge: str, func_attr_mapping: Callable
):
    # input: path
    # output: a mapping of each node/edge to its attr, and each node to its global-idx if exists
    assert node_or_edge in {"node", "edge", "graph"}
    dict_map = {"discrete": {}, "continuous": {}, "embed": {}}
    attr_shuffle = config["semantics"].get("attr_shuffle", False)

    discrete_attr = config["semantics"][node_or_edge]["discrete"]
    world_identifier = config["attr_world_identifier"]
    if discrete_attr is not None:
        share_vocab = config["semantics"][node_or_edge].get("share_vocab", False)
        ignored_val = config["semantics"][node_or_edge]["ignored_val"]
        tmp_map = func_attr_mapping(path, data, discrete_attr)
        dict_map["discrete"] = (
            {
                k: _tokenize_discrete_attr(
                    v.astype(str),
                    world_identifier,
                    node_or_edge,
                    ignored_val,
                    attr_shuffle,
                    share_vocab=share_vocab,
                )
                for k, v in tmp_map.items()
            }
            if tmp_map
            else {(-1, -1): None}
        )

    continuous_attr = config["semantics"][node_or_edge]["continuous"]
    if continuous_attr is not None:
        assert (
            discrete_attr is None
        ), "Supporting both discrete and continuous attr is NOT implemented yet!"
        ignored_val = config["semantics"][node_or_edge]["ignored_val"]
        tmp_map = func_attr_mapping(path, data, continuous_attr)
        dict_map["continuous"] = {
            k: _tokenize_continuous_attr(
                v.astype(str), world_identifier, node_or_edge, ignored_val, attr_shuffle
            )
            for k, v in tmp_map.items()
        }

    embed_attr = config["semantics"][node_or_edge].get("embed", None)
    if embed_attr is not None:
        tmp_map = func_attr_mapping(path, data, embed_attr)
        dict_map["embed"] = (
            {k: v.tolist() for k, v in tmp_map.items()} if tmp_map else {(-1, -1): None}
        )
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


def mask_semantics_attr(data: Data, config: Dict, node_or_edge: str):
    # input: path
    # output: a mapping of each node/edge to its attr, and each node to its global-idx if exists
    assert node_or_edge in {"node", "edge", "graph"}

    discrete_attr = config["semantics"][node_or_edge]["discrete"]
    if discrete_attr is not None:
        data[discrete_attr] = data[discrete_attr] * 0

    continuous_attr = config["semantics"][node_or_edge]["continuous"]
    if continuous_attr is not None:
        assert (
            discrete_attr is None
        ), "Supporting both discrete and continuous attr is NOT implemented yet!"
        data[continuous_attr] = data[continuous_attr] * 0
    return data


def mask_semantics_raw_node_edge_attr(data: Data, config: Dict):
    data = data.clone()
    data = mask_semantics_attr(data, config, "node")
    data = mask_semantics_attr(data, config, "edge")
    data = mask_semantics_attr(data, config, "graph")
    return data
