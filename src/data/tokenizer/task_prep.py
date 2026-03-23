"""Task-specific input preparation functions.

Each function is registered via ``@_inputs_deco("task_type")`` so that
``prepare_inputs_for_task(task_type, in_dict, **kwargs)`` dispatches to
the correct handler.
"""

import math
import random
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data

from ...utils import control_flow
from .masking import (
    _mask_input_ids,
    _mask_stacked_input_ids_v2,
    _pad_stacked_targets,
    get_mask_of_raw_seq,
)
from .types import MOL_ENERGY_BIN_LEN, MOL_ENERGY_SCALE

_inputs_deco = control_flow.Register()
prepare_inputs_for_task = _inputs_deco.build  # return func results
get_inputs_preparation_func = _inputs_deco.get  # return the func


# ---------------------------------------------------------------------------
# Pretrain tasks
# ---------------------------------------------------------------------------


@_inputs_deco("pretrain")
def prepare_inputs_for_pretrain(in_dict, **kwargs):
    in_dict["split_lens"] = [len(in_dict["input_ids"])]
    in_dict["attn_modes"] = ["full"]
    return in_dict


@_inputs_deco("pretrain-cl")
@_inputs_deco("pretrain-mlm")
def prepare_inputs_for_pretrain_mlm(
    in_dict, *, graph: Data, gtokenizer, ls_len: List[int], **kwargs
):
    # add eos to input_ids
    add_eos = True
    if add_eos:
        input_ids = in_dict["input_ids"] + in_dict["labels"][-1:]  # add eos
        len_extended_tokens = 1
    else:
        input_ids = in_dict["input_ids"]
        len_extended_tokens = 0
    if len(gtokenizer.config.get("ensemble_datasets", [])) >= 2:
        assert gtokenizer.mpe is None, "NOT implemented for packed token sequence"
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

    # Only update ls_len[-1] if NOT using packed sequences (mpe is None)
    # For packed sequences, ls_len already accounts for the EOS token from pack_token_seq
    if gtokenizer.mpe is None:
        ls_len[-1] = ls_len[-1] + len_extended_tokens

    # 1. set-up parameters for SMTP: scheduled masked token prediction
    mask_token_id = gtokenizer.get_mask_token_id()
    pad_token_id = gtokenizer.pad_token_id
    assert mask_token_id != pad_token_id
    conf = gtokenizer.config["pretrain_mlm"]
    assert conf["name"] in {"polynomial", "cosine", "fixed"}
    if conf["name"] == "fixed":
        alpha_t = conf["params"]["fixed_ratio"]
    elif conf["name"] == "polynomial":
        # 3-> cubic, 2-> square, 1-> linear, 0.5-> sqrt
        powers = conf["params"]["power"]
        umr_min, umr_max = gtokenizer.train_cfg.pretrain_mlm.params.umr_clip
        assert 0 <= umr_min <= umr_max <= 1
        r = random.random()
        t = umr_min + (umr_max - umr_min) * r  # rescale to [mr_min, mr_max]
        alpha_t = 1 - t**powers
        alpha_t_prime = -powers * t ** (powers - 1)
        # Fig. 1 @ https://arxiv.org/pdf/2406.04329
        wgt = powers / t  # - alpha_t_prime / (1 - alpha_t)
        if gtokenizer.train_cfg.pretrain_mlm.dlm_wgt:
            in_dict["wgt"] = wgt
    else:
        alpha_t = math.cos(random.random() * math.pi) * 0.5 + 0.5
    mask_token_precent = conf["params"]["mtp"]
    all_vocab_ids = gtokenizer.get_all_vocab_ids()
    # 2. mask input_ids and generate corresponding labels for training
    new_input_ids, new_labels_mask = [], []
    idx_left = 0
    for idx_right in ls_len:
        _input_ids = input_ids[idx_left:idx_right]
        idx_left = idx_right
        curr_mask_ratio = alpha_t
        if (gtokenizer.mpe is not None) and (idx_right > gtokenizer.mpe):
            curr_mask_ratio = 0
        if isinstance(input_ids[0], Iterable):
            if add_eos:
                last_token_id = _input_ids[-1][0]
                assert (
                    last_token_id == gtokenizer.get_eos_token_id()
                ), f"{last_token_id}!={gtokenizer.get_eos_token_id()}\nls_len:{ls_len}\nidx_right:{idx_right},\ninput_ids:{input_ids}\n_input_ids:{_input_ids}"
            _input_ids, _labels_mask = _mask_stacked_input_ids_v2(
                _input_ids,
                mask_token_id,
                all_vocab_ids,
                curr_mask_ratio,
                mask_token_precent=mask_token_precent,
                pad_token_id=pad_token_id,
                has_eos=add_eos,
                stack_method=gtokenizer.stack_method,
            )
        else:
            assert isinstance(input_ids[0], int)
            last_token_id = _input_ids[-1]
            assert (
                last_token_id == gtokenizer.get_eos_token_id()
            ), f"{last_token_id}!={gtokenizer.get_eos_token_id()}\nls_len:{ls_len}\nidx_right:{idx_right},\ninput_ids:{input_ids}\n_input_ids:{_input_ids}"
            _input_ids, _labels_mask = _mask_input_ids(
                _input_ids,
                mask_token_id,
                all_vocab_ids,
                curr_mask_ratio,
                mask_token_precent,
                pad_token_id,
            )
        new_input_ids.extend(_input_ids)
        new_labels_mask.extend(_labels_mask)
    input_ids, labels_mask = new_input_ids, new_labels_mask
    if hasattr(gtokenizer, "stack_method") and gtokenizer.stack_method == "long":
        node_attr_dim = gtokenizer.config["semantics"]["node"]["dim"]
        labels_mask = [
            _pad_stacked_targets(
                i, ls_labels, node_attr_dim=node_attr_dim, padding_val=-100
            )
            for i, ls_labels in enumerate(labels_mask)
        ]

    if gtokenizer.train_cfg.task_type == "pretrain-cl":
        input_ids, labels_mask, len_extended_tokens = _add_gsum_tokens_for_cl(
            input_ids, labels_mask, gtokenizer, len_extended_tokens
        )
    in_dict["input_ids"] = input_ids
    in_dict["labels"] = labels_mask
    if gtokenizer.mpe is None:
        in_dict["attention_mask"].extend([1] * len_extended_tokens)
        in_dict["split_lens"] = [len(in_dict["input_ids"])]
        in_dict["attn_modes"] = ["full"]
    else:
        lens = (np.array(ls_len) - np.array([0] + ls_len[:-1])).tolist()
        in_dict["attention_mask"].extend([1] * len_extended_tokens)
        in_dict["split_lens"] = [int(l) for l in lens]
        pad_len = gtokenizer.mpe - sum(lens)  # TODO: currently `split_lens` and `sample_lens` are almost the same, shall be changed in the future
        in_dict["sample_lens"] = list(in_dict["split_lens"]) + [pad_len]
        in_dict["attn_modes"] = ["full"] * len(lens)
        new_pos = []
        for l in lens:
            new_pos.extend(range(int(l)))
        in_dict["position_ids"] = new_pos
    if "embed" in in_dict:
        dim = len(in_dict["embed"][0])
        extended_embed = np.zeros((len_extended_tokens, dim), dtype=np.float32).tolist()
        in_dict["embed"].extend(extended_embed)
        assert len(in_dict["embed"]) == len(
            in_dict["input_ids"]
        ), f"{len(in_dict['embed'])} != {len(in_dict['input_ids'])}"
    return in_dict


def _add_gsum_tokens_for_cl(input_ids, labels_mask, gtokenizer, len_extended_tokens):
    """Add gsum tokens for contrastive learning, to avoid multiple use of <eos> token"""
    special_token_id = gtokenizer.get_gsum_token_id()
    ls_extend_tokens = [special_token_id]
    inputs_instance = input_ids[0]
    if isinstance(inputs_instance, List):
        ls_extend_tokens = [
            [token_id] * len(inputs_instance) for token_id in ls_extend_tokens
        ]
    input_ids.extend(ls_extend_tokens)
    label_pad_token_id = gtokenizer.label_pad_token_id
    ls_extend_tokens = [label_pad_token_id]
    labels_mask_instance = labels_mask[0]
    if isinstance(labels_mask_instance, List):
        ls_extend_tokens = [
            [token_id] * len(inputs_instance) for token_id in ls_extend_tokens
        ]
    labels_mask.extend(ls_extend_tokens)
    len_extended_tokens = len_extended_tokens + len(ls_extend_tokens)
    return input_ids, labels_mask, len_extended_tokens


@_inputs_deco("pretrain-smtp")
@_inputs_deco("pretrain-coord-cl")
@_inputs_deco("pretrain-coord")
def prepare_inputs_for_pretrain_coord(
    in_dict, *, graph: Data, gtokenizer, ls_raw_node_idx: List[int], **kwargs
):
    input_ids = in_dict["input_ids"] + in_dict["labels"][-1:]  # add eos
    len_extended_tokens = 1
    assert len(gtokenizer.config.get("ensemble_datasets", [])) == 0
    assert gtokenizer.mpe is None
    in_dict["input_ids"] = input_ids
    in_dict["attention_mask"].extend([1] * len_extended_tokens)
    if "embed" in in_dict:
        dim = len(in_dict["embed"][0])
        extended_embed = np.zeros((len_extended_tokens, dim), dtype=np.float32).tolist()
        in_dict["embed"].extend(extended_embed)
        assert len(in_dict["embed"]) == len(
            in_dict["input_ids"]
        ), f"{len(in_dict['embed'])} != {len(in_dict['input_ids'])}"
    input_ids = _attach_node_mask_to_inputs(
        ls_raw_node_idx,
        len_extended_tokens,
        in_dict["input_ids"],
    )
    in_dict["input_ids"] = input_ids.tolist()
    in_dict["split_lens"] = [len(in_dict["input_ids"])]
    in_dict["attn_modes"] = ["full"]
    return in_dict


@_inputs_deco("pretrain-mlm-coord")
def prepare_inputs_for_pretrain_mlm_coord(
    in_dict,
    *,
    graph: Data,
    gtokenizer,
    ls_raw_node_idx: List[int],
    ls_len: List[int],
    **kwargs,
):
    in_dict = prepare_inputs_for_pretrain_mlm(
        in_dict,
        graph=graph,
        gtokenizer=gtokenizer,
        ls_len=ls_len,
    )
    input_ids = _attach_node_mask_to_inputs(
        ls_raw_node_idx,
        len_extended_tokens=1,
        input_ids=in_dict["input_ids"],
    )
    in_dict["input_ids"] = input_ids.tolist()
    return in_dict


def _attach_node_mask_to_inputs(ls_raw_node_idx, len_extended_tokens, input_ids):
    ls_raw_node_idx.extend([-1] * len_extended_tokens)
    node_idx = np.array(ls_raw_node_idx) + 1
    node_idx_clip = np.clip(node_idx, 0, 4)
    node_mask = get_mask_of_raw_seq(node_idx, mask_type="random")
    node_mask = node_mask * (node_idx > 0)
    edge_seq = list(zip([0] + node_idx.tolist()[:-1], node_idx.tolist()))
    edge_mask = get_mask_of_raw_seq(edge_seq, mask_type="random")
    edge_mask = edge_mask * (np.array(edge_seq) > 0).all(axis=-1)
    node_type = np.vstack([node_idx_clip, node_mask, node_idx, edge_mask]).T
    input_ids = np.hstack([np.array(input_ids), node_type])
    return input_ids


@_inputs_deco("pretrain-ltp")
def prepare_inputs_for_last_token_pred_in_pretrain(in_dict, **kwargs):
    raw_labels = in_dict["labels"]
    in_dict["labels"] = [-100] * (len(raw_labels) - 1) + raw_labels[-1:]
    in_dict["split_lens"] = [len(in_dict["input_ids"])]
    in_dict["attn_modes"] = ["causal"]
    return in_dict


@_inputs_deco("pretrain-euler")
def prepare_inputs_for_euler_pretrain(in_dict, *, gtokenizer, **kwargs):
    eos_token_id = gtokenizer.get_eos_token_id()
    label_pad_token_id = gtokenizer.label_pad_token_id
    raw_labels = in_dict["labels"]
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
    in_dict["split_lens"] = [len(in_dict["input_ids"])]
    in_dict["attn_modes"] = ["causal"]
    return in_dict


# ---------------------------------------------------------------------------
# Supervised tasks
# ---------------------------------------------------------------------------


@_inputs_deco("graph")
def prepare_inputs_for_graph_lvl_task(
    in_dict: Dict[str, List[Union[int, Iterable[int]]]],
    *,
    graph: Data,
    gtokenizer,
    ls_raw_node_idx: List[int],
    **kwargs,
):
    ls_extend_tokens = []
    bin_labels = None
    in_dict = _extend_input_dict(in_dict, ls_extend_tokens)
    in_dict["graph_labels"] = torch.squeeze(graph.y).tolist()
    in_dict["labels"] = bin_labels or in_dict["labels"]

    len_extended_tokens = len(ls_extend_tokens)
    if "embed" in in_dict:
        dim = len(in_dict["embed"][0])
        extended_embed = np.zeros((len_extended_tokens, dim), dtype=np.float32).tolist()
        in_dict["embed"].extend(extended_embed)
        assert len(in_dict["embed"]) == len(
            in_dict["input_ids"]
        ), f"{len(in_dict['embed'])} != {len(in_dict['input_ids'])}"
    if ls_raw_node_idx is not None:
        input_ids = _attach_node_mask_to_inputs(
            ls_raw_node_idx,
            len_extended_tokens,
            in_dict["input_ids"],
        )
        in_dict["input_ids"] = input_ids.tolist()
    in_dict["split_lens"] = [len(in_dict["input_ids"])]
    in_dict["attn_modes"] = ["full"]
    return in_dict


@_inputs_deco("edge")
def prepare_inputs_for_edge_lvl_task(
    in_dict: Dict[str, List[Union[int, List[int]]]],
    *,
    graph: Data,
    gtokenizer,
    tgt_edge_src_token_id: Union[int, Tuple, List],
    tgt_edge_dst_token_id: Union[int, Tuple, List],
    tgt_edge_attr_token_id: Union[Tuple[int], List[int]],
    **kwargs,
):
    ls_src_dst = [tgt_edge_src_token_id, tgt_edge_dst_token_id]
    if not tgt_edge_attr_token_id:
        random.shuffle(ls_src_dst)
    if isinstance(tgt_edge_dst_token_id, Tuple) or isinstance(
        tgt_edge_dst_token_id, List
    ):
        ls_src_dst = [item for row in ls_src_dst for item in row]
    raw_ls_extend_tokens = list(ls_src_dst)
    ls_extend_tokens = list(ls_src_dst)
    ls_extend_emb = []
    if isinstance(in_dict["input_ids"][0], List):
        dict_mapping = {x[0]: x for x in in_dict["input_ids"]}
        ls_extend_tokens = [list(dict_mapping[x]) for x in raw_ls_extend_tokens]
        edge_dim = gtokenizer.config["semantics"]["edge"]["dim"]
        if edge_dim > 0:
            assert len(ls_extend_tokens) == 2
            default_edge_attr_id = gtokenizer.get_default_edge_attr_id()
            assert len(default_edge_attr_id) == edge_dim
            ls_extend_tokens[0] = ls_extend_tokens[0][:-edge_dim] + list(
                default_edge_attr_id
            )
            if tgt_edge_attr_token_id:
                assert len(tgt_edge_attr_token_id) == edge_dim
                ls_extend_tokens[1] = ls_extend_tokens[1][:-edge_dim] + list(
                    tgt_edge_attr_token_id
                )
            else:
                ls_extend_tokens[1] = ls_extend_tokens[1][:-edge_dim] + list(
                    default_edge_attr_id
                )
        if "embed" in in_dict:
            assert len(in_dict["input_ids"]) == len(in_dict["embed"])
            dict_emb_mapping = {
                x[0]: y for x, y in zip(in_dict["input_ids"], in_dict["embed"])
            }
            ls_extend_emb = [list(dict_emb_mapping[x]) for x in raw_ls_extend_tokens]
    in_dict = _extend_input_dict(in_dict, ls_extend_tokens)
    in_dict["idx"] = (
        graph.seed_node.tolist() if hasattr(graph, "seed_node") else ls_src_dst
    )
    in_dict["edge_labels"] = graph.y.item()
    if "embed" in in_dict:
        in_dict["embed"].extend(ls_extend_emb)
        assert len(in_dict["input_ids"]) == len(in_dict["embed"])
    if hasattr(graph, "wgt"):
        in_dict["wgt"] = graph.wgt.item()
    in_dict["split_lens"] = [len(in_dict["input_ids"])]
    in_dict["attn_modes"] = ["full"]
    return in_dict


@_inputs_deco("node")
def prepare_inputs_for_node_lvl_task(
    in_dict: Dict[str, List[Union[int, List[int]]]],
    *,
    graph: Data,
    gtokenizer,
    eos_token_id: int,
    tgt_node_token_id: Union[int, Tuple],
    **kwargs,
):
    if isinstance(tgt_node_token_id, int):
        ls_token_ids = [tgt_node_token_id]
    else:
        ls_token_ids = list(tgt_node_token_id)
    raw_ls_extend_tokens = list(ls_token_ids)
    ls_extend_tokens = list(ls_token_ids)
    ls_extend_emb = []
    if isinstance(in_dict["input_ids"][0], List):
        dict_mapping = {x[0]: x for x in in_dict["input_ids"]}
        ls_extend_tokens = [list(dict_mapping[x]) for x in raw_ls_extend_tokens]
        edge_dim = gtokenizer.config["semantics"]["edge"]["dim"]
        if edge_dim > 0:
            assert len(ls_extend_tokens) == 1
            default_edge_attr_id = gtokenizer.get_default_edge_attr_id()
            assert len(default_edge_attr_id) == edge_dim
            ls_extend_tokens[0] = ls_extend_tokens[0][:-edge_dim] + list(
                default_edge_attr_id
            )
        if "embed" in in_dict:
            assert len(in_dict["input_ids"]) == len(in_dict["embed"])
            dict_emb_mapping = {
                x[0]: y for x, y in zip(in_dict["input_ids"], in_dict["embed"])
            }
            ls_extend_emb = [list(dict_emb_mapping[x]) for x in raw_ls_extend_tokens]
    in_dict = _extend_input_dict(in_dict, ls_extend_tokens)
    in_dict["idx"] = ls_token_ids
    assert graph.num_nodes == graph.y.shape[0]
    in_dict["node_labels"] = graph.y[graph.root_n_id].tolist()
    if "embed" in in_dict:
        in_dict["embed"].extend(ls_extend_emb)
        assert len(in_dict["input_ids"]) == len(in_dict["embed"])
    if hasattr(graph, "wgt"):
        in_dict["wgt"] = graph.wgt.item()
    in_dict["split_lens"] = [len(in_dict["input_ids"])]
    in_dict["attn_modes"] = ["full"]
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
    permute_label: bool = True,
    **kwargs,
):
    if (
        hasattr(graph, "y")
        and (graph.y is not None)
        and (graph.y.shape[0] == graph.num_nodes)
    ):
        nodev2_labels = graph.y[:, 0].tolist()
    else:
        nodev2_labels = [-100] * graph.x.shape[0]
    assert len(tgt_node_token_id) == len(nodev2_labels)
    mapping = dict(zip(tgt_node_token_id, nodev2_labels))
    mapping2raw_node_idx = dict(zip(tgt_node_token_id, list(range(len(nodev2_labels)))))
    if isinstance(in_dict["input_ids"][0], int):
        in_dict["nodev2_labels"] = [
            mapping.pop(ele, -100) for ele in in_dict["input_ids"]
        ]
        in_dict["raw_node_idx"] = [
            mapping2raw_node_idx.pop(ele, -100) for ele in in_dict["input_ids"]
        ]
    else:
        in_dict["nodev2_labels"] = [
            mapping.pop(ele[0], -100) for ele in in_dict["input_ids"]
        ]
        in_dict["raw_node_idx"] = [
            mapping2raw_node_idx.pop(ele[0], -100) for ele in in_dict["input_ids"]
        ]
    if loss_type == "token_ce_intra":
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
            keys=("nodev2_labels", "raw_node_idx"),
            vals=(-100, -100),
        )
    in_dict["split_lens"] = [len(in_dict["input_ids"])]
    in_dict["attn_modes"] = ["full"]
    return in_dict


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _extend_input_dict(
    in_dict, ls_extend_tokens, keys=tuple(), vals=tuple()
):
    len_extended_tokens = len(ls_extend_tokens)
    inputs_instance = in_dict["input_ids"][0]
    if isinstance(inputs_instance, List):
        ls_extend_tokens = [
            [token_id] * len(inputs_instance) if isinstance(token_id, int) else token_id
            for token_id in ls_extend_tokens
        ]
    in_dict["input_ids"].extend(ls_extend_tokens)
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


# Unused but kept for backward compatibility
def prepare_inputs_for_oneid_a2c_pred_as_node_pred(in_dict, **kwargs):
    raw_labels = in_dict["labels"]
    in_dict["node_labels"] = (
        raw_labels[-1] - 19
    )  # 19 is vocab-id of euler path start node
    assert in_dict["node_labels"] < 200, f"{in_dict['node_labels']} > 200"
    return in_dict
