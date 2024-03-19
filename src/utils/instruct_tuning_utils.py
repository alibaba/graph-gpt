from typing import Dict
from torch_geometric.data import Data

from . import control_flow

_instruct = control_flow.Register()
get_instruct = _instruct.build  # return func results


def follow_instructions(
    graph: Data,
    tokenization_config: Dict,
    node_structure_mapping: Dict,
    edge_structure_mapping: Dict,
    node_semantics_mapping: Dict,
    edge_semantics_mapping: Dict,
    gtokenizer=None,
):
    ls = []
    instruct_conf = tokenization_config["semantics"].get("instructions", {})
    if instruct_conf.get("enable", False):
        for func in instruct_conf["func"]:
            if func["valid"]:
                name = func["name"] if gtokenizer is None else func["name"] + "-stack"
                tmp_ls = get_instruct(
                    name,
                    graph,
                    node_structure_mapping=node_structure_mapping,
                    edge_structure_mapping=edge_structure_mapping,
                    node_semantics_mapping=node_semantics_mapping,
                    edge_semantics_mapping=edge_semantics_mapping,
                    config=tokenization_config,
                    gtokenizer=gtokenizer,
                )
                ls.extend(tmp_ls)
    return ls


@_instruct("a2d")
def _obtain_acc2device(
    graph: Data, *, node_structure_mapping: Dict, config: Dict, **kwargs
):
    # oneid task
    reserved_token_id = graph.key_type.item() if hasattr(graph, "key_type") else 0

    a2d_tokens = graph.a2d.tolist()  # [N, 2]
    a2d_tokens = _flatten_list(a2d_tokens)

    reindexed_a2d = [node_structure_mapping[ele] for ele in a2d_tokens]
    reindexed_a2d = _flatten_list(reindexed_a2d)

    instruct_tokens = [
        config["semantics"]["common"]["reserved_token"][reserved_token_id]
    ]
    return instruct_tokens + reindexed_a2d


@_instruct("a2d-stack")
def _obtain_stacked_acc2device(
    graph: Data,
    *,
    gtokenizer,
    node_structure_mapping: Dict,
    node_semantics_mapping: Dict,
    edge_semantics_mapping: Dict,
    config: Dict,
    **kwargs
):
    # oneid task
    reserved_token_id = graph.key_type.item() if hasattr(graph, "key_type") else 0

    a2d_tokens = graph.a2d.tolist()  # [N, 2]
    a2d_tokens = _flatten_list(a2d_tokens)

    ls_tokens = [
        _get_all_node_feats(
            node,
            (-1, -1),
            node_structure_mapping,
            node_semantics_mapping,
            edge_semantics_mapping,
        )
        for node in a2d_tokens
    ]

    instruct_tokens = [
        config["semantics"]["common"]["reserved_token"][reserved_token_id]
    ] * len(ls_tokens[0])
    # instruct_tokens_feat = _get_all_node_feats(-1, (-1,-1), node_structure_mapping, node_semantics_mapping, edge_semantics_mapping)
    # instruct_tokens = instruct_tokens + instruct_tokens_feat[1:]
    return [instruct_tokens] + ls_tokens


def _flatten_list(ls):
    return [ele for sub_ls in ls for ele in sub_ls]


def _get_all_node_feats(
    node,
    edge,
    node_structure_mapping,
    node_semantics_mapping,
    edge_semantics_mapping,
    edge_semantics_default=None,
):
    ls_node_id = list(node_structure_mapping[node])
    if node_semantics_mapping["discrete"]:
        ls_node_attr = node_semantics_mapping["discrete"][node]
    else:
        ls_node_attr = []
    if edge_semantics_mapping["discrete"]:
        ls_edge_attr = edge_semantics_mapping["discrete"].get(
            edge, edge_semantics_default
        )
    else:
        ls_edge_attr = []
    # For jump-edge, no edge-attr available, so use default edge attr
    # TODO: implement removing edge-type, e.g., removing edge-bi
    return ls_node_id + ls_node_attr + ls_edge_attr
