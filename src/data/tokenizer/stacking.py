"""Stacking strategies for StackedGSTTokenizer."""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from ...utils import instruct_tuning_utils
from .graph_encoding import _tokenize_discrete_attr


def add_eos_embed(ls_embed):
    if ls_embed:
        ls = [0.0] * len(ls_embed[0])
        ls_embed.append(ls)
    return ls_embed


def stack_node_edge_graph_attr_to_node(
    gtokenizer,
    path: List[Tuple[int, int]],
    node_structure_mapping,
    edge_structure_mapping,
    node_semantics_mapping,
    edge_semantics_mapping,
    graph_semantics_mapping,
):
    ls_tokens = []  # For next/masked-token-prediction
    ls_embed = []  # Embed features as input only
    ls_raw_node_idx = []  # raw node-idx for 3D position labeling

    # 1. work on 1st node in the path
    # 1.1 For discrete feature as tokens
    if path:
        node, _ = path[0]
    else:  # For graph with single node, path == []
        node = 0
    edge = (-1, -1)
    ls = instruct_tuning_utils._get_all_node_feats(
        node,
        edge,
        node_structure_mapping=node_structure_mapping,
        edge_structure_mapping=edge_structure_mapping,
        node_semantics_mapping=node_semantics_mapping,
        edge_semantics_mapping=edge_semantics_mapping,
    )
    ls_tokens.append(ls)

    # 1.2 For embed features
    ls_e = instruct_tuning_utils._get_all_node_feats(
        node,
        edge,
        node_semantics_mapping=node_semantics_mapping,
        edge_semantics_mapping=edge_semantics_mapping,
        edge_semantics_default=gtokenizer.get_default_edge_embed(),
        attr_type="embed",
    )
    ls_embed.append(ls_e)

    # 1.3 For raw node-idx
    ls_raw_node_idx.append(node)

    # 2. work on subsequent edges & nodes in the path
    for edge in path:
        _, node = edge
        # 2.1 For discrete feature as tokens
        ls = instruct_tuning_utils._get_all_node_feats(
            node,
            edge,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            edge_semantics_default=gtokenizer.get_default_edge_attr(),
        )
        ls_tokens.append(ls)
        # 2.2 For embed feature
        ls_e = instruct_tuning_utils._get_all_node_feats(
            node,
            edge,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            edge_semantics_default=gtokenizer.get_default_edge_embed(),
            attr_type="embed",
        )
        ls_embed.append(ls_e)
        # 2.3 For raw node-idx
        ls_raw_node_idx.append(node)
    return ls_tokens, ls_embed, ls_raw_node_idx


def stack_attr_to_node_and_edge(
    gtokenizer,
    path: List[Tuple[int, int]],
    node_structure_mapping,
    edge_structure_mapping,
    node_semantics_mapping,
    edge_semantics_mapping,
    graph_semantics_mapping,
):
    ls_tokens = []  # For next/masked-token-prediction
    ls_embed = []  # Embed features as input only
    ls_raw_node_idx = []  # raw node-idx for 3D position labeling

    # 1. work on 1st node in the path
    # 1.1 For discrete feature as tokens
    if path:
        node, _ = path[0]
    else:  # For graph with single node, path == []
        node = 0
    edge = (-1, -1)
    ls = instruct_tuning_utils._get_all_node_feats(
        node,
        edge,
        node_structure_mapping=node_structure_mapping,
        edge_structure_mapping=edge_structure_mapping,
        node_semantics_mapping=node_semantics_mapping,
        edge_semantics_mapping=edge_semantics_mapping,
    )
    ls_tokens.append(ls)

    # 1.2 For embed features
    ls_e = instruct_tuning_utils._get_all_node_feats(
        node,
        edge,
        node_semantics_mapping=node_semantics_mapping,
        edge_semantics_mapping=edge_semantics_mapping,
        attr_type="embed",
    )
    ls_embed.append(ls_e)
    pad_embed = tuple([0] * len(ls_e))

    # 1.3 For raw node-idx
    ls_raw_node_idx.append(node)

    for edge in path:
        # 2. obtain ls-tokens/embeds/node_idx from `edge`
        # 2.1 discrete features
        node = -1
        ls = instruct_tuning_utils._get_all_node_feats(
            node,
            edge,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            node_semantics_default=gtokenizer.get_default_node_attr(),
            edge_semantics_default=gtokenizer.get_default_edge_attr(),
        )
        ls_tokens.append(ls)
        # 2.2 embed features
        ls_embed.append(list(pad_embed))
        # 2.3 raw node-idx
        ls_raw_node_idx.append(node)

        # 3. obtain ls-tokens/embeds/node_idx from `node`
        _, node = edge
        edge = (-1, -1)
        # 3.1 discrete fatures
        ls = instruct_tuning_utils._get_all_node_feats(
            node,
            edge,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            node_semantics_default=gtokenizer.get_default_node_attr(),
            edge_semantics_default=gtokenizer.get_default_edge_attr(),
        )
        ls_tokens.append(ls)
        # 3.2 embed features
        ls_e = instruct_tuning_utils._get_all_node_feats(
            node,
            edge,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            attr_type="embed",
        )
        ls_embed.append(ls_e)
        # 3.3 raw node-idx
        ls_raw_node_idx.append(node)
    return ls_tokens, ls_embed, ls_raw_node_idx


def get_default_semantics_attr_mapping(graph: Data, config: Dict, node_or_edge: str):
    assert node_or_edge in {"node", "edge", "graph"}

    discrete_attr = config["semantics"][node_or_edge]["discrete"]
    share_vocab = config["semantics"][node_or_edge].get("share_vocab", False)
    world_identifier = config["attr_world_identifier"]
    ls_tokens = []
    if discrete_attr is not None:
        assert (
            len(graph[discrete_attr].shape) == 2
        ), f"{len(graph[discrete_attr].shape)} != 2"
        dim = graph[discrete_attr].shape[1]
        raw_attr = torch.zeros((1, dim)).to(graph[discrete_attr])
        raw_attr = raw_attr[0].numpy().astype(str)
        ls_tokens = _tokenize_discrete_attr(
            raw_attr,
            world_identifier,
            node_or_edge,
            remove_val=True,
            share_vocab=share_vocab,
        )
    return ls_tokens


def get_default_semantics_embed_mapping(graph: Data, config: Dict, node_or_edge: str):
    assert node_or_edge in {"node", "edge", "graph"}

    embed_attr = config["semantics"][node_or_edge].get("embed", None)
    default_embed = None
    if embed_attr is not None:
        assert len(graph[embed_attr].shape) == 2, f"{len(graph[embed_attr].shape)} != 2"
        dim = graph[embed_attr].shape[1]
        raw_attr = torch.zeros((1, dim)).to(graph[embed_attr])
        default_embed = np.zeros_like(raw_attr[0].numpy()).tolist()
    return default_embed
