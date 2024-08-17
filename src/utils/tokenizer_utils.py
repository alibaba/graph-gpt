import random
import math

import numpy as np
import torch
from typing import List, Union, Tuple, Dict, Iterable
from torch_geometric.data import Data
from . import control_flow

TASK_TYPES = {
    "pretrain",
    "pretrain-mlm",
    "pretrain-ltp",
    "pretrain-euler",
    "node",
    "nodev2",
    "edge",
    "graph",
    "graph-dh",
    "graph-ddh",
}
ATTR_ASSIGNMENT_TYPES = {"first", "last", "random", "all", "mix"}

_inputs_deco = control_flow.Register()
prepare_inputs_for_task = _inputs_deco.build  # return func results
get_inputs_preparation_func = _inputs_deco.get  # return the func


@_inputs_deco("pretrain")
def prepare_inputs_for_pretrain(in_dict, **kwargs):
    return in_dict


def _mask_ids(
    mask_token_id,
    global_rnd_id,
    raw_id,
    mask_token_precent: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    pad_token_id: int = 0,
):
    # If the i-th token is chosen, we replace the i-th token with
    # (1) the [MASK] token 80% of the time
    # (2) a random token 10% of the time
    # (3) the unchanged i-th token 10% of the time
    rate_vec = np.cumsum(mask_token_precent)
    assert rate_vec[2] == 1.0, f"rate_vec: {rate_vec}"
    if raw_id == pad_token_id:
        return raw_id
    rnd = random.random()
    if rnd < rate_vec[0]:
        return mask_token_id
    elif rnd < rate_vec[1]:
        return global_rnd_id
    else:
        return raw_id


def _get_keys(idx, ls: List[int], ls_of_ls: List[List[int]]):
    if idx % 2 == 0:  # key for node
        key = ls[0]
    else:  # key for edge
        prev_node = ls_of_ls[idx - 1][0]
        next_node = ls_of_ls[idx + 1][0]
        key = (
            (prev_node, next_node) if prev_node < next_node else (next_node, prev_node)
        )
    return key


def _mask_stacked_input_ids(
    input_ids: List[List[int]],
    mask_token_id,
    all_vocab_ids,
    mask_ratio: float = 0.15,
    mask_token_precent: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    pad_token_id: int = 0,
    has_eos: bool = True,
    stack_method: str = "short",
):
    # labels_mask = [[-100] * len(input_ids[0])] * len(input_ids)
    # above list's element are the same list, causing problems when indexing each element
    labels_mask = np.full((len(input_ids), len(input_ids[0])), -100).tolist()
    if has_eos:
        eos = input_ids[-1:]
        input_ids = input_ids[:-1]
    else:
        eos = []
        input_ids = input_ids
    if stack_method == "short":
        keys = [ele[0] for ele in input_ids]
    else:
        assert (
            len(input_ids) % 2 == 1
        ), f"tmp_ids: {input_ids},\nhas_eos: {has_eos},\n{locals()}"
        keys = [_get_keys(i, ele, input_ids) for i, ele in enumerate(input_ids)]
    keys_set = set(keys)
    keys_masked = random.sample(
        list(keys_set), k=int(round(len(keys_set) * mask_ratio))
    )
    keys_masked_set = set(keys_masked)
    # print(f"keys: {keys}\nkeys_masked_set: {keys_masked_set}")

    for idx, (ls_tokens, key) in enumerate(zip(input_ids, keys)):
        if key in keys_masked_set:
            labels_mask[idx] = input_ids[idx]
            input_ids[idx] = [
                _mask_ids(
                    mask_token_id,
                    random.sample(all_vocab_ids, k=1)[0],
                    ele,
                    mask_token_precent,
                    pad_token_id,
                )
                for ele in ls_tokens
            ]
    input_ids.extend(eos)
    return input_ids, labels_mask


def _mask_stacked_input_ids_v2(
    input_ids: List[List[int]],
    mask_token_id,
    all_vocab_ids,
    mask_ratio: float = 0.15,
    mask_token_precent: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    pad_token_id: int = 0,
    has_eos: bool = True,
    stack_method: str = "short",
):
    # v2 choose tokens to mask globally
    seq = len(input_ids)
    dim = len(input_ids[0])
    input_ids = np.array(input_ids)
    labels_mask = np.full((seq, dim), -100)

    indices = list(np.ndindex((seq, dim)))
    idx_masked = random.sample(
        range(len(indices)), k=int(round(len(indices) * mask_ratio))
    )
    rate_vec = np.cumsum(mask_token_precent)

    for idx in idx_masked:
        idx_seq, idx_dim = indices[idx]
        labels_mask[idx_seq, idx_dim] = input_ids[idx_seq, idx_dim]
        # If the i-th token is chosen, we replace the i-th token with
        # (1) the [MASK] token 80% of the time
        # (2) a random token 10% of the time
        # (3) the unchanged i-th token 10% of the time
        if input_ids[idx_seq, idx_dim] != pad_token_id:
            rnd = random.random()
            if rnd < rate_vec[0]:
                input_ids[idx_seq, idx_dim] = mask_token_id
            elif rnd < rate_vec[1]:
                input_ids[idx_seq, idx_dim] = random.sample(all_vocab_ids, k=1)[0]
            else:
                pass
    return input_ids.tolist(), labels_mask.tolist()


def _mask_input_ids(
    input_ids: List[int],
    mask_token_id,
    all_vocab_ids,
    mask_ratio: float = 0.15,
    mask_token_precent: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    pad_token_id: int = 0,
):
    labels_mask = [-100] * len(input_ids)
    idx_masked = random.sample(
        range(len(input_ids)), k=int(round(len(input_ids) * mask_ratio))
    )
    rate_vec = np.cumsum(mask_token_precent)
    assert rate_vec[2] == 1.0, f"rate_vec: {rate_vec}"
    for idx in idx_masked:
        labels_mask[idx] = input_ids[idx]
        # If the i-th token is chosen, we replace the i-th token with
        # (1) the [MASK] token 80% of the time
        # (2) a random token 10% of the time
        # (3) the unchanged i-th token 10% of the time
        if input_ids[idx] != pad_token_id:
            rnd = random.random()
            if rnd < rate_vec[0]:
                input_ids[idx] = mask_token_id
            elif rnd < rate_vec[1]:
                input_ids[idx] = random.sample(all_vocab_ids, k=1)[0]
            else:
                pass
    return input_ids, labels_mask


def _pad_stacked_targets(i, ls_token_ids, node_attr_dim=9, padding_val=-100):
    if i % 2 == 0:  # pad node labels
        ls_token_ids = [
            token_id if j <= node_attr_dim else padding_val
            for j, token_id in enumerate(ls_token_ids)
        ]
    else:  # pad edge labels
        ls_token_ids = [
            token_id if j > node_attr_dim else padding_val
            for j, token_id in enumerate(ls_token_ids)
        ]
    return ls_token_ids


@_inputs_deco("pretrain-mlm")
def prepare_inputs_for_pretrain(in_dict, *, graph: Data, gtokenizer, **kwargs):
    # add eos to input_ids
    add_eos = True
    input_ids = in_dict["input_ids"] + in_dict["labels"][-1:]  # add eos
    len_extended_tokens = 1
    if len(gtokenizer.config["ensemble_datasets"]) >= 2:
        reserved_semantics_token = gtokenizer.get_common_semantics()[graph.idx_of_ds]
        token_id = gtokenizer._map_tokens_to_ids(reserved_semantics_token)
        ls_extend_tokens = [token_id]
        inputs_instance = input_ids[0]
        if isinstance(inputs_instance, List):
            ls_extend_tokens = [
                [token_id] * len(inputs_instance) for token_id in ls_extend_tokens
            ]
        input_ids.extend(ls_extend_tokens)
        len_extended_tokens += len(ls_extend_tokens)

    mask_token_id = gtokenizer.get_mask_token_id()
    pad_token_id = gtokenizer.pad_token_id
    assert mask_token_id != pad_token_id
    conf = gtokenizer.config["pretrain_mlm"]
    assert conf["name"] in {"polynomial", "cosine", "fixed"}
    if conf["name"] == "fixed":
        mask_ratio = conf["params"]["fixed_ratio"]
    elif conf["name"] == "polynomial":
        # 3-> cubic, 2-> square, 1-> linear, 0.5-> sqrt
        powers = conf["params"]["power"]
        mask_ratio = 1 - random.random() ** powers
    else:
        mask_ratio = min(
            max(math.cos(random.random() * math.pi / 2), 0), 1
        )  # MaskGIT-cos
    mask_token_precent = conf["params"]["mtp"]
    # [MASK] token 80% of the time, i.e., (0.8, 0.1, 0.1) -> BERT paper
    all_vocab_ids = gtokenizer.get_all_vocab_ids()
    if isinstance(input_ids[0], Iterable):
        input_ids, labels_mask = _mask_stacked_input_ids(
            input_ids,
            mask_token_id,
            all_vocab_ids,
            mask_ratio,
            mask_token_precent=mask_token_precent,
            pad_token_id=pad_token_id,
            has_eos=add_eos,
            stack_method=gtokenizer.stack_method,
        )
    else:
        assert isinstance(input_ids[0], int)
        input_ids, labels_mask = _mask_input_ids(
            input_ids,
            mask_token_id,
            all_vocab_ids,
            mask_ratio,
            mask_token_precent,
            pad_token_id,
        )
    if hasattr(gtokenizer, "stack_method") and gtokenizer.stack_method == "long":
        node_attr_dim = gtokenizer.config["semantics"]["node"]["dim"]
        labels_mask = [
            _pad_stacked_targets(
                i, ls_labels, node_attr_dim=node_attr_dim, padding_val=-100
            )
            for i, ls_labels in enumerate(labels_mask)
        ]

    in_dict["input_ids"] = input_ids
    in_dict["labels"] = labels_mask
    in_dict["position_ids"].extend(
        list(
            range(
                len(in_dict["position_ids"]),
                len(in_dict["position_ids"]) + len_extended_tokens,
            )
        )
    )
    in_dict["attention_mask"].extend([1] * len_extended_tokens)
    return in_dict


@_inputs_deco("pretrain-ltp")
def prepare_inputs_for_last_token_pred_in_pretrain(in_dict, **kwargs):
    raw_labels = in_dict["labels"]
    in_dict["labels"] = [-100] * (len(raw_labels) - 1) + raw_labels[-1:]
    return in_dict


@_inputs_deco("pretrain-euler")
def prepare_inputs_for_last_token_pred_in_pretrain(in_dict, *, gtokenizer, **kwargs):
    eos_token_id = gtokenizer.get_eos_token_id()
    label_pad_token_id = gtokenizer.label_pad_token_id
    raw_labels = in_dict["labels"]
    # eos_idx = raw_labels.index(sep_token_id)
    # in_dict["labels"] = [-100] * (eos_idx + 2) + raw_labels[eos_idx + 2 :]
    # above v1 commented, for pack==0 only
    # below v2 for pack==0/1
    flag = 0
    new_labels = [-100] * len(raw_labels)
    for i in range(2, len(raw_labels)):
        if (raw_labels[i - 1] == label_pad_token_id) and (
            raw_labels[i - 2] == label_pad_token_id
        ):
            flag = 1
        if raw_labels[i - 1] == eos_token_id:
            flag = 0
        if flag == 1:
            new_labels[i] = raw_labels[i]
    in_dict["labels"] = new_labels
    return in_dict


# @_inputs_deco("node")
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
    gtokenizer,
    is_training: bool = None,
    **kwargs,
):
    # reserved_semantics_token = gtokenizer.get_common_semantics()[graph.idx_of_ds]
    # token_id = gtokenizer._map_tokens_to_ids(reserved_semantics_token)
    # ls_extend_tokens = [token_id]
    # ABOVE for fine-tuning with train data from two different source, e.g., PCQM4M-v2 & CEPDB
    ls_extend_tokens = []
    in_dict = _extend_input_dict(in_dict, ls_extend_tokens, cmpe=gtokenizer.cmpe)
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


@_inputs_deco("node")
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


@_inputs_deco("nodev2")
def prepare_inputs_for_node_v2_token_lvl_task(
    in_dict: Dict[str, List[Union[int, List[int]]]],
    *,
    graph: Data,
    gtokenizer,
    tgt_node_token_id: Union[int, Tuple],
    num_labels: int = 10,
    loss_type: str = "token_ce",
    **kwargs,
):
    assert hasattr(graph, "y") and graph.y.shape[0] == graph.num_nodes
    nodev2_labels = graph.y[:, 0].tolist()
    assert len(tgt_node_token_id) == len(nodev2_labels)
    permute_label = True
    # if permute_label:  # NOT WORKING
    #     ls = list(range(num_labels))
    #     random.shuffle(ls)
    #     nodev2_labels = [ls[x] if x != -100 else -100 for x in nodev2_labels]
    mapping = dict(zip(tgt_node_token_id, nodev2_labels))
    mapping2raw_node_idx = dict(zip(tgt_node_token_id, list(range(len(nodev2_labels)))))
    if isinstance(in_dict["input_ids"][0], int):
        in_dict["nodev2_labels"] = [
            mapping.pop(ele, -100) for ele in in_dict["input_ids"]
        ]  # replace `.get` with `.pop` to ensure one node only trained once
        in_dict["raw_node_idx"] = [
            mapping2raw_node_idx.pop(ele, -100) for ele in in_dict["input_ids"]
        ]  # replace `.get` with `.pop` to ensure evaluation is applied on every node only once!
    else:
        in_dict["nodev2_labels"] = [
            mapping.pop(ele[0], -100) for ele in in_dict["input_ids"]
        ]
        in_dict["raw_node_idx"] = [
            mapping2raw_node_idx.pop(ele[0], -100) for ele in in_dict["input_ids"]
        ]
    if loss_type == "token_ce_intra":
        # below is to apply intra-instance clustering/classification
        reserved_semantics_tokens = gtokenizer.get_common_semantics()
        assert (
            len(reserved_semantics_tokens) >= num_labels
        ), f"len(reserved_semantics_tokens)=={len(reserved_semantics_tokens)} < num_labels=={num_labels}"
        if permute_label:
            random.shuffle(reserved_semantics_tokens)
        in_dict["cls_idx"] = [len(in_dict["input_ids"])]
        ls_extend_tokens = [
            gtokenizer._map_tokens_to_ids(x) for x in reserved_semantics_tokens
        ]
        in_dict = _extend_input_dict(
            in_dict,
            ls_extend_tokens,
            cmpe=gtokenizer.cmpe,
            keys=("nodev2_labels", "raw_node_idx"),
            vals=(-100, -100),
        )
    return in_dict


def _extend_input_dict(
    in_dict, ls_extend_tokens, cmpe=int(1e8), keys=tuple(), vals=tuple()
):
    len_extended_tokens = len(ls_extend_tokens)
    inputs_instance = in_dict["input_ids"][0]
    if isinstance(inputs_instance, List):
        ls_extend_tokens = [
            [token_id] * len(inputs_instance) for token_id in ls_extend_tokens
        ]
    in_dict["input_ids"].extend(ls_extend_tokens)
    in_dict["position_ids"].extend(
        list(
            [
                ele % cmpe
                for ele in range(
                    in_dict["position_ids"][-1] + 1,
                    in_dict["position_ids"][-1] + 1 + len_extended_tokens,
                )
            ]
        )
    )
    labels_instance = in_dict["labels"][0]
    if isinstance(labels_instance, List):
        ls_extend_labels = [[-100] * len(labels_instance)] * len_extended_tokens
    else:
        ls_extend_labels = [-100] * len_extended_tokens
    in_dict["labels"].extend(ls_extend_labels)
    in_dict["attention_mask"].extend([1] * len_extended_tokens)
    for key, val in zip(keys, vals):
        in_dict[key].extend([val] * len_extended_tokens)
    return in_dict
