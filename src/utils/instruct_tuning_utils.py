from typing import List, Dict
from copy import deepcopy
import random
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
    ls_labels = []
    instruct_conf = tokenization_config["semantics"].get("instructions", {})
    if instruct_conf.get("enable", False):
        for func in instruct_conf["func"]:
            if func["valid"]:
                name = (
                    func["name"]
                    if gtokenizer.__class__.__name__ != "StackedGSTTokenizer"
                    else func["name"] + "-stack"
                )
                kwargs = deepcopy(func)
                kwargs.pop("valid")
                kwargs.pop("name")
                tmp_ls, tmp_ls_labels = get_instruct(
                    name,
                    graph,
                    node_structure_mapping=node_structure_mapping,
                    edge_structure_mapping=edge_structure_mapping,
                    node_semantics_mapping=node_semantics_mapping,
                    edge_semantics_mapping=edge_semantics_mapping,
                    config=tokenization_config,
                    gtokenizer=gtokenizer,
                    **kwargs,
                )
                ls.extend(tmp_ls)
                ls_labels.extend(tmp_ls_labels)
    return ls, ls_labels


@_instruct("homo_lumo")
def _obtain_homolumo_properties(
    graph: Data,
    *,
    node_structure_mapping: Dict,
    config: Dict,
    gtokenizer,
    mask_ratio: float = 0,
    **kwargs,
):
    # molecule task
    reserved_token_id = 0

    vals = graph.y.numpy().astype(str).tolist()  # [1, 1]
    if (vals[0][0] != "nan") and (random.random() < 1 - mask_ratio):
        val_tokens = [f"<{x}>" for x in list(vals[0][0])]

        instruct_tokens = [
            config["semantics"]["common"]["reserved_token"][reserved_token_id]
        ]
        ls = instruct_tokens + val_tokens
        ls_labels = ls[1:] + [gtokenizer.get_eos_token()]
        return ls, ls_labels
    else:
        return [], []


@_instruct("cepdb_prop_all")
def _obtain_all_cepdb_properties(
    graph: Data, *, node_structure_mapping: Dict, config: Dict, gtokenizer, **kwargs
):
    # molecule task, 7 cepdb's properties
    # "mass", "pce", "voc", "jsc", "e_homo_alpha", "e_gap_alpha", "e_lumo_alpha"
    if graph.y.size()[1] == 7:
        ls = []
        vals = graph.y.numpy().astype(str).tolist()  # [1, 7]
        for reserved_token_id, val in enumerate(vals[0]):
            val_tokens = [f"<{x}>" for x in list(val)]

            instruct_tokens = [
                config["semantics"]["common"]["reserved_token"][reserved_token_id]
            ]
            tmp_ls = instruct_tokens + val_tokens
            ls.append(tmp_ls)
        random.shuffle(ls)
        ls_labels = [x[1:] + [gtokenizer.get_eos_token()] for x in ls]
        return _flatten_list(ls), _flatten_list(ls_labels)
    else:
        return [], []


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
    **kwargs,
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
    node_structure_mapping: Dict = None,
    edge_structure_mapping: Dict = None,
    node_semantics_mapping: Dict = None,
    edge_semantics_mapping: Dict = None,
    node_semantics_default: List = None,
    edge_semantics_default: List = None,
    attr_type: str = "discrete",
):
    assert attr_type in {"discrete", "embed"}
    ls_node_id = (
        list(node_structure_mapping[node]) if node_structure_mapping is not None else []
    )
    if node_semantics_mapping[attr_type]:
        ls_node_attr = node_semantics_mapping[attr_type][node]
    else:
        ls_node_attr = []

    ls_edge_struct = (
        [edge_structure_mapping[edge]] if edge_structure_mapping is not None else []
    )
    if edge_semantics_mapping[attr_type]:
        ls_edge_attr = edge_semantics_mapping[attr_type].get(
            edge, edge_semantics_default
        )  # `get` is for the case of jump-edge
    else:
        ls_edge_attr = []
    # For jump-edge, no edge-attr available, so use default edge attr
    # TODO: implement removing edge-type, e.g., removing edge-bi
    return ls_node_id + ls_node_attr + ls_edge_struct + ls_edge_attr
