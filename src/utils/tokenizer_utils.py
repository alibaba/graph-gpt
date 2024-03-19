import random
import torch
from typing import List, Union, Tuple, Dict, Iterable
from torch_geometric.data import Data
from . import control_flow

TASK_TYPES = {
    "pretrain",
    "pretrain-ltp",
    "pretrain-euler",
    "node",
    "edge",
    "graph",
    "graph-dh",
    "graph-ddh",
}
ATTR_ASSIGNMENT_TYPES = {"first", "last", "random", "all"}

_inputs_deco = control_flow.Register()
prepare_inputs_for_task = _inputs_deco.build  # return func results
get_inputs_preparation_func = _inputs_deco.get  # return the func


@_inputs_deco("pretrain")
def prepare_inputs_for_pretrain(in_dict, **kwargs):
    return in_dict


@_inputs_deco("pretrain-ltp")
def prepare_inputs_for_last_token_pred_in_pretrain(in_dict, **kwargs):
    raw_labels = in_dict["labels"]
    in_dict["labels"] = [-100] * (len(raw_labels) - 1) + raw_labels[-1:]
    return in_dict


@_inputs_deco("pretrain-euler")
def prepare_inputs_for_last_token_pred_in_pretrain(in_dict, **kwargs):
    sep_token_id = 7
    eos_token_id = 18
    raw_labels = in_dict["labels"]
    # eos_idx = raw_labels.index(sep_token_id)
    # in_dict["labels"] = [-100] * (eos_idx + 2) + raw_labels[eos_idx + 2 :]
    # above v1 commented, for pack==0 only
    # below v2 for pack==0/1
    flag = 0
    new_labels = [-100] * len(raw_labels)
    for i in range(2, len(raw_labels)):
        if raw_labels[i - 2] == sep_token_id:
            flag = 1
        if raw_labels[i - 1] == eos_token_id:
            flag = 0
        if flag == 1:
            new_labels[i] = raw_labels[i]
    in_dict["labels"] = new_labels
    return in_dict


@_inputs_deco("node")
def prepare_inputs_for_oneid_a2c_pred_as_node_pred(in_dict, **kwargs):
    raw_labels = in_dict["labels"]
    in_dict["node_labels"] = (
        raw_labels[-1] - 19
    )  # 19 is vocab-id of euler path start node
    assert in_dict["node_labels"] < 200, f"{in_dict['node_labels']} > 200"
    return in_dict


@_inputs_deco("graph-dh")
def prepare_inputs_for_graph_lvl_double_head_task(
    in_dict: Dict[str, List[int]],
    *,
    graph: Data,
    eos_token_id: int,
    gsum_token_id: int,
    tgt_pos: torch.Tensor,
    gtokenizer,
    **kwargs,
):
    """
    inputs Graph-level DoubleHead tasks:
        task1 -> unidirectional NTP task
        task2 -> bidirectional regression task, e.g., predict 3d coordinates (noises)
    :param in_dict:
    :param graph:
    :param eos_token_id:
    :param gsum_token_id:
    :param tgt_pos: [num_nodes, 3]
    :param gtokenizer:
    :param kwargs:
    :return:
    """
    assert gsum_token_id is not None
    num_nodes = graph.x.shape[0] if torch.abs(tgt_pos).sum() > 1e-8 else 0
    assert (
        num_nodes <= gtokenizer.config["structure"]["node"]["scope_base"]
    ), "NOT Implemented"
    assert (tgt_pos.shape[0] == num_nodes) or (num_nodes == 0)
    # 1. add node-id as extended tokens for 3d position regression task
    ls_node_tokens = [str(x) for x in range(num_nodes)]
    ls_node_tokens_id = [gtokenizer.vocab_map[x] for x in ls_node_tokens]
    # ls_extend_tokens = [eos_token_id, gsum_token_id]
    ls_extend_tokens = [gsum_token_id] + ls_node_tokens_id
    # 2. based on the extended tokens-id, reformat the input dict elements
    len_extended_tokens = len(ls_extend_tokens)
    in_dict["input_ids"].extend(ls_extend_tokens)
    in_dict["position_ids"].extend(
        list(
            range(
                len(in_dict["position_ids"]),
                len(in_dict["position_ids"]) + len_extended_tokens,
            )
        )
    )
    in_dict["labels"].extend([-100] * len_extended_tokens)
    in_dict["attention_mask"].extend([1] * len_extended_tokens)
    in_dict["graph_labels"] = torch.squeeze(graph.y).tolist()
    # 2.1 attention_mask_bi for the 3d position regression task
    seq = len(in_dict["attention_mask"])
    in_dict["attention_mask_bi"] = [0 if i < seq - num_nodes else 1 for i in range(seq)]
    # 2.2 pad the 3d position tensor: [N,3] -> [N+1+euler-seq-len,3]
    if torch.abs(tgt_pos).sum() > 1e-8:
        pos_pad = torch.zeros(seq - num_nodes, 3, dtype=torch.float32)
        in_dict["pos"] = torch.cat([pos_pad, tgt_pos], dim=0)
    else:
        in_dict["pos"] = torch.zeros(seq, 3, dtype=torch.float32)
    return in_dict


@_inputs_deco("graph-ddh")
def prepare_inputs_for_graph_lvl_denoising_double_head_task(
    in_dict: Dict[str, List[int]],
    *,
    graph: Data,
    eos_token_id: int,
    gsum_token_id: int,
    tgt_pos: torch.Tensor,
    gtokenizer,
    **kwargs,
):
    """
    inputs Graph-level DenoisingDoubleHead tasks:
        task1 -> unidirectional NTP task
        task2 -> bidirectional regression task, e.g., predict 3d coordinates noises
    :param in_dict:
    :param graph:
    :param eos_token_id:
    :param gsum_token_id:
    :param tgt_pos: [num_nodes, 3]
    :param gtokenizer:
    :param kwargs:
    :return:
    """
    assert gsum_token_id is not None
    num_nodes = graph.x.shape[0] if torch.abs(tgt_pos).sum() > 1e-8 else 0
    assert (
        num_nodes <= gtokenizer.config["structure"]["node"]["scope_base"]
    ), "NOT Implemented"
    assert (tgt_pos.shape[0] == num_nodes) or (num_nodes == 0)
    # 1. add node-id as extended tokens for 3d position regression task
    ls_node_tokens = [str(x) for x in range(num_nodes)]
    ls_node_tokens_id = [gtokenizer.vocab_map[x] for x in ls_node_tokens]
    # ls_extend_tokens = [eos_token_id, gsum_token_id]
    ls_extend_tokens = [gsum_token_id] + ls_node_tokens_id
    # 2. based on the extended tokens-id, reformat the input dict elements
    len_extended_tokens = len(ls_extend_tokens)
    in_dict["input_ids"].extend(ls_extend_tokens)
    in_dict["position_ids"].extend(
        list(
            range(
                len(in_dict["position_ids"]),
                len(in_dict["position_ids"]) + len_extended_tokens,
            )
        )
    )
    in_dict["labels"].extend([-100] * len_extended_tokens)
    in_dict["attention_mask"].extend([1] * len_extended_tokens)
    in_dict["graph_labels"] = torch.squeeze(graph.y).tolist()
    # 2.1 attention_mask_bi for the 3d position regression task
    seq = len(in_dict["attention_mask"])
    in_dict["attention_mask_bi"] = [0 if i < seq - num_nodes else 1 for i in range(seq)]
    # 2.2 pad the 3d position tensor: [N,3] -> [N+1+euler-seq-len,3]
    if torch.abs(tgt_pos).sum() > 1e-8:
        pos_pad = torch.zeros(seq - num_nodes, 3, dtype=torch.float32)
        in_dict["pos"] = torch.cat([pos_pad, tgt_pos], dim=0)
    else:
        in_dict["pos"] = torch.zeros(seq, 3, dtype=torch.float32)
    # 2.3 add moise to coordinates, and prepare denosing labels
    ori_pos = in_dict["pos"]
    noise = torch.randn(ori_pos.shape).to(ori_pos) * gtokenizer.config["semantics"].get(
        "noise_scale", 0.2
    )
    in_dict["pos"] = ori_pos + noise
    in_dict["noise"] = noise
    return in_dict


@_inputs_deco("graph")
def prepare_inputs_for_graph_lvl_task(
    in_dict: Dict[str, List[Union[int, Iterable[int]]]],
    *,
    graph: Data,
    eos_token_id: int,
    gsum_token_id: int,
    **kwargs,
):
    assert gsum_token_id is not None
    # ls_extend_tokens = [eos_token_id, gsum_token_id]
    ls_extend_tokens = [gsum_token_id]  # to be compatible with lf's best result setting
    len_extended_tokens = len(ls_extend_tokens)
    inputs_instance = in_dict["input_ids"][0]
    if isinstance(inputs_instance, List):
        ls_extend_tokens = [
            [token_id] * len(inputs_instance) for token_id in ls_extend_tokens
        ]
    in_dict["input_ids"].extend(ls_extend_tokens)
    in_dict["position_ids"].extend(
        list(
            range(
                len(in_dict["position_ids"]),
                len(in_dict["position_ids"]) + len_extended_tokens,
            )
        )
    )
    in_dict["labels"].extend([-100] * len_extended_tokens)
    in_dict["attention_mask"].extend([1] * len_extended_tokens)
    in_dict["graph_labels"] = torch.squeeze(graph.y).tolist()
    return in_dict


@_inputs_deco("edge")
def prepare_inputs_for_edge_lvl_task(
    in_dict: Dict[str, List[int]],
    *,
    graph: Data,
    eos_token_id: int,
    tgt_edge_src_token_id: Union[int, Tuple, List],
    tgt_edge_dst_token_id: Union[int, Tuple, List],
    **kwargs,
):
    ls_src_dst = [tgt_edge_src_token_id, tgt_edge_dst_token_id]
    random.shuffle(ls_src_dst)
    if isinstance(tgt_edge_dst_token_id, Tuple) or isinstance(
        tgt_edge_dst_token_id, List
    ):
        ls_src_dst = [item for row in ls_src_dst for item in row]
    extended_token_len = len(ls_src_dst) + 1  # +1 for eos token
    in_dict["idx"] = (
        graph.seed_node.tolist() if hasattr(graph, "seed_node") else ls_src_dst
    )
    in_dict["input_ids"].extend([eos_token_id] + ls_src_dst)
    in_dict["position_ids"].extend(
        list(
            range(
                len(in_dict["position_ids"]),
                len(in_dict["position_ids"]) + extended_token_len,
            )
        )
    )
    in_dict["labels"].extend([-100] * extended_token_len)
    in_dict["attention_mask"].extend([1] * extended_token_len)
    in_dict["edge_labels"] = graph.y.item()
    return in_dict


# @_inputs_deco("node")
def prepare_inputs_for_node_lvl_task(
    in_dict: Dict[str, List[int]],
    *,
    graph: Data,
    eos_token_id: int,
    tgt_node_token_id: Union[int, Tuple],
    **kwargs,
):
    if isinstance(tgt_node_token_id, int):
        ls_token_ids = [tgt_node_token_id]
    else:
        # for node identity encoding with multiple tokens
        ls_token_ids = list(tgt_node_token_id)
    extended_token_len = len(ls_token_ids) + 1  # +1 for eos token
    in_dict["idx"] = ls_token_ids
    in_dict["input_ids"].extend([eos_token_id] + ls_token_ids)
    in_dict["position_ids"].extend(
        [
            len(in_dict["position_ids"]),
            len(in_dict["position_ids"]) + extended_token_len,
        ]
    )
    in_dict["labels"].extend([-100] * extended_token_len)
    in_dict["attention_mask"].extend([1] * extended_token_len)
    assert graph.num_nodes == graph.y.shape[0]
    in_dict["node_labels"] = graph.y[graph.root_n_id].tolist()
    return in_dict
