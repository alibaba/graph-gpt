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


def _get_node_feats_batch(
    node: int,
    edge: Tuple[int, int],
    node_structure_mapping: Dict,
    edge_structure_mapping: Dict,
    node_semantics_mapping: Dict,
    edge_semantics_mapping: Dict,
    node_semantics_default: List = None,
    edge_semantics_default: List = None,
    node_embed_default: List = None,
    edge_embed_default: List = None,
) -> Tuple[List, List]:
    """
    Batch extraction of both discrete tokens and embed features for a node-edge pair.
    Returns (discrete_tokens, embed_features) tuple.
    """
    # Discrete tokens
    ls_node_id = (
        [node_structure_mapping[node]]
        if node_structure_mapping is not None
        else []  # List[str]
    )
    if node_semantics_mapping.get("discrete"):
        ls_node_attr = (
            node_semantics_mapping["discrete"].get(node, node_semantics_default) or []
        )
    else:
        ls_node_attr = []

    ls_edge_struct = (
        [edge_structure_mapping[edge]] if edge_structure_mapping is not None else []
    )
    if edge_semantics_mapping.get("discrete"):
        ls_edge_attr = (
            edge_semantics_mapping["discrete"].get(edge, edge_semantics_default) or []
        )
    else:
        ls_edge_attr = []

    discrete_tokens = ls_node_id + ls_node_attr + ls_edge_struct + ls_edge_attr

    # Embed features
    if node_semantics_mapping.get("embed"):
        embed_feats = (
            node_semantics_mapping["embed"].get(node, node_embed_default) or []
        )
    else:
        embed_feats = []

    return discrete_tokens, embed_feats


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

    # Cache default values
    default_edge_attr = gtokenizer.get_default_edge_attr()
    default_edge_embed = gtokenizer.get_default_edge_embed()

    # Pre-extract all nodes from path
    if path:
        nodes_in_path = [path[0][0]]  # First node
        nodes_in_path.extend(tgt for _, tgt in path)  # Subsequent nodes
        edges_in_path = [(-1, -1)] + list(
            path
        )  # First edge is default, then path edges
    else:
        nodes_in_path = [0]
        edges_in_path = [(-1, -1)]

    # Batch process all node-edge pairs
    for node, edge in zip(nodes_in_path, edges_in_path):
        discrete_tokens, embed_feats = _get_node_feats_batch(
            node,
            edge,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            edge_semantics_default=default_edge_attr,
            edge_embed_default=default_edge_embed,
        )
        ls_tokens.append(discrete_tokens)
        ls_embed.append(embed_feats)
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

    # Cache default values
    default_node_attr = gtokenizer.get_default_node_attr()
    default_edge_attr = gtokenizer.get_default_edge_attr()

    # 1. work on 1st node in the path
    if path:
        node, _ = path[0]
    else:  # For graph with single node, path == []
        node = 0
    edge = (-1, -1)

    discrete_tokens, embed_feats = _get_node_feats_batch(
        node,
        edge,
        node_structure_mapping=node_structure_mapping,
        edge_structure_mapping=edge_structure_mapping,
        node_semantics_mapping=node_semantics_mapping,
        edge_semantics_mapping=edge_semantics_mapping,
    )
    ls_tokens.append(discrete_tokens)
    ls_embed.append(embed_feats)
    pad_embed = [0.0] * len(embed_feats) if embed_feats else []
    ls_raw_node_idx.append(node)

    # 2. Process edges and subsequent nodes
    for edge in path:
        # 2. Edge row (node=-1)
        node = -1
        discrete_tokens, embed_feats = _get_node_feats_batch(
            node,
            edge,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            node_semantics_default=default_node_attr,
            edge_semantics_default=default_edge_attr,
        )
        ls_tokens.append(discrete_tokens)
        ls_embed.append(list(pad_embed))  # No embed for edge rows
        ls_raw_node_idx.append(node)

        # 3. Node row
        _, node = edge
        edge = (-1, -1)
        discrete_tokens, embed_feats = _get_node_feats_batch(
            node,
            edge,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            node_semantics_default=default_node_attr,
            edge_semantics_default=default_edge_attr,
        )
        ls_tokens.append(discrete_tokens)
        ls_embed.append(embed_feats)
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
