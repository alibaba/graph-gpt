"""Graph-to-token encoding: attribute tokenization and semantics mapping."""

import random
from typing import Callable, Dict, List

from torch_geometric.data import Data


class DigitTokenCache:
    """Cache for digit tokens like <0>, <1>, etc."""

    _cache: Dict[str, str] = {}

    @classmethod
    def get_digit_token(cls, digit: str) -> str:
        """Get cached digit token like <0>, <1>, etc."""
        if digit in cls._cache:
            return cls._cache[digit]
        token = f"<{digit}>"
        cls._cache[digit] = token
        return token


class TokenCache:
    """Global cache for token strings to avoid repeated string formatting."""

    _cache: Dict[tuple, str] = {}
    _hits = 0
    _misses = 0

    @classmethod
    def get_token(
        cls, world_id: str, node_edge: str, col_idx: int, col_val: str
    ) -> str:
        """Get cached token or create new one."""
        key = (world_id, node_edge, col_idx, col_val)
        if key in cls._cache:
            cls._hits += 1
            return cls._cache[key]
        cls._misses += 1
        token = f"{world_id}#{node_edge}#{col_idx}#{col_val}"
        cls._cache[key] = token
        return token

    @classmethod
    def get_token_share_vocab(
        cls, world_id: str, node_edge: str, col_idx: int, col_val: str
    ) -> str:
        """Get cached token with shared vocab format."""
        key = (world_id, node_edge, -1, col_val)
        if key in cls._cache:
            cls._hits += 1
            return cls._cache[key]
        cls._misses += 1
        token = f"{world_id}#{node_edge}#-1#{col_val}"
        cls._cache[key] = token
        return token

    @classmethod
    def get_token_no_val(cls, world_id: str, node_edge: str, col_idx: int) -> str:
        """Get cached token without value."""
        key = (world_id, node_edge, col_idx, None)
        if key in cls._cache:
            cls._hits += 1
            return cls._cache[key]
        cls._misses += 1
        token = f"{world_id}#{node_edge}#{col_idx}"
        cls._cache[key] = token
        return token

    @classmethod
    def get_token_no_val_share_vocab(cls, world_id: str, node_edge: str) -> str:
        """Get cached token without value and shared vocab."""
        key = (world_id, node_edge, -1, None)
        if key in cls._cache:
            cls._hits += 1
            return cls._cache[key]
        cls._misses += 1
        token = f"{world_id}#{node_edge}#-1"
        cls._cache[key] = token
        return token

    @classmethod
    def get_identifier_token(cls, world_id: str, node_edge: str, col_idx: int) -> str:
        """Get cached identifier token for continuous attrs."""
        key = (world_id, node_edge, col_idx, "identifier")
        if key in cls._cache:
            cls._hits += 1
            return cls._cache[key]
        cls._misses += 1
        token = f"{world_id}#{node_edge}#{col_idx}#1"
        cls._cache[key] = token
        return token

    @classmethod
    def get_stats(cls):
        return {"hits": cls._hits, "misses": cls._misses, "cache_size": len(cls._cache)}


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
        if share_vocab:
            token = TokenCache.get_token_no_val_share_vocab(
                world_identifier, node_edge_identifier
            )
            tokens = [token] * len(raw_attr)
        else:
            tokens = [
                TokenCache.get_token_no_val(
                    world_identifier, node_edge_identifier, col_idx
                )
                for col_idx in range(len(raw_attr))
            ]
    else:
        ignored_str = str(ignored_val) if ignored_val is not None else None
        if share_vocab:
            tokens = [
                TokenCache.get_token_share_vocab(
                    world_identifier, node_edge_identifier, col_idx, col_val
                )
                for col_idx, col_val in enumerate(raw_attr)
                if col_val != ignored_str
            ]
        else:
            tokens = [
                TokenCache.get_token(
                    world_identifier, node_edge_identifier, col_idx, col_val
                )
                for col_idx, col_val in enumerate(raw_attr)
                if col_val != ignored_str
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
    ignored_str = str(ignored_val) if ignored_val is not None else None
    is_graph = node_edge_identifier == "graph"

    def _process_each_col(col_idx, col_val):
        # Use cached digit tokens
        ls_col_val = list(col_val)
        ls_col_val = _remove_lead_zero(ls_col_val)
        ls_col_val = [DigitTokenCache.get_digit_token(x) for x in ls_col_val]

        if is_graph:
            identifier = "<gsum>"
        else:
            identifier = TokenCache.get_identifier_token(
                world_identifier, node_edge_identifier, col_idx
            )
        return [identifier] + ls_col_val

    tokens = [
        _process_each_col(col_idx, col_val)
        for col_idx, col_val in enumerate(raw_attr)
        if col_val != ignored_str
    ]
    if shuffle:
        random.shuffle(tokens)
    return tokens


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
    # Build edge→index mapping once
    edge_index_map = {}
    for i, (s, t) in enumerate(
        zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())
    ):
        edge_index_map[(s, t)] = i

    tmp_map = {}
    for src, tgt in path:
        idx = edge_index_map.get((src, tgt)) or edge_index_map.get((tgt, src))
        if idx is not None:
            tmp_map[(src, tgt)] = data[attr_name][idx].numpy()
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
