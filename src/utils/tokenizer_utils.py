import random
import math
from scipy.linalg import block_diag
import numpy as np
import torch
from typing import List, Union, Tuple, Dict, Iterable, Optional
from torch_geometric.data import Data
from dataclasses import dataclass
from transformers.utils import ModelOutput
from . import control_flow

TASK_TYPES = {
    "pretrain",
    "pretrain-smtp",
    "pretrain-mlm",
    "pretrain-mlm-coord",
    "pretrain-coord",
    "pretrain-ltp",
    "pretrain-euler",
    "pretrain-cl",
    "pretrain-coord-cl",
    "node",
    "nodev2",
    "edge",
    "graph",
}
ATTR_ASSIGNMENT_TYPES = {"first", "last", "random", "all", "mix"}
MOL_ENERGY_BIN_LEN = 16
MOL_ENERGY_SCALE = 1000

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
        list(keys_set), k=int(np.ceil(len(keys_set) * mask_ratio))
    )  # use `np.ceil` to ensure at least one token is masked!!
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
    mask_token_precent: Tuple[float, float, float] = (1, 0, 0),
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
        range(len(indices)), k=int(np.ceil(len(indices) * mask_ratio))
    )
    rate_vec = np.cumsum(mask_token_precent)

    for idx in idx_masked:
        idx_seq, idx_dim = indices[idx]
        labels_mask[idx_seq, idx_dim] = input_ids[idx_seq, idx_dim]
        # If the i-th token is chosen, we replace the i-th token with (Below BERT settings)
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
        range(len(input_ids)), k=int(np.ceil(len(input_ids) * mask_ratio))
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


def _pad_stacked_targets(
    i, ls_token_ids, *, node_attr_dim=9, padding_val=-100, eos_token_id=None
):
    if i % 2 == 0:  # pad node labels
        ls_token_ids = [
            token_id if j <= node_attr_dim else padding_val
            for j, token_id in enumerate(ls_token_ids)
        ]
    else:  # pad edge labels
        ls_token_ids = [
            token_id if (j > node_attr_dim or token_id == eos_token_id) else padding_val
            for j, token_id in enumerate(ls_token_ids)
        ]
    return ls_token_ids


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

    ls_len[-1] = ls_len[-1] + len_extended_tokens

    # 1. set-up parameters for SMTP: scheduled masked token prediction
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
        # mask_ratio = min(
        #     max(math.cos(random.random() * math.pi / 2), 0), 1
        # )  # MaskGIT-cos
        mask_ratio = math.cos(random.random() * math.pi) * 0.5 + 0.5
    mask_token_precent = conf["params"]["mtp"]
    # [MASK] token 80% of the time, i.e., (0.8, 0.1, 0.1) -> BERT paper
    all_vocab_ids = gtokenizer.get_all_vocab_ids()
    # 2. mask input_ids and generate corresponding labels for training
    new_input_ids, new_labels_mask = [], []
    idx_left = 0
    for idx_right in ls_len:
        # the `for` loop is for packing several sequences together in pre-training
        _input_ids = input_ids[idx_left:idx_right]
        idx_left = idx_right
        curr_mask_ratio = mask_ratio
        if (gtokenizer.mpe is not None) and (idx_right > gtokenizer.mpe):
            # in case of pack_tokens and smtp-pretrain, if the last seq beyond mpe, do NOT mask,
            # therefore NOT make prediction -> DEPRECATED because make result worse
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
    if gtokenizer.mpe is None:
        in_dict["attention_mask"].extend([1] * len_extended_tokens)
    else:
        # create block-wise bi-attention for packed sequences
        lens = np.array(ls_len) - np.array([0] + ls_len[:-1])
        attns = [np.ones([each_len, each_len], dtype=int) for each_len in lens]
        in_dict["attention_mask"] = block_diag(*attns)
    if "embed" in in_dict:
        dim = len(in_dict["embed"][0])
        extended_embed = np.zeros((len_extended_tokens, dim), dtype=np.float32).tolist()
        in_dict["embed"].extend(extended_embed)
        assert len(in_dict["embed"]) == len(
            in_dict["input_ids"]
        ), f"{len(in_dict['embed'])} != {len(in_dict['input_ids'])}"
    return in_dict


@_inputs_deco("pretrain-smtp")
@_inputs_deco("pretrain-coord-cl")
@_inputs_deco("pretrain-coord")
def prepare_inputs_for_pretrain_coord(
    in_dict, *, graph: Data, gtokenizer, ls_raw_node_idx: List[int], **kwargs
):
    # `pretrain-smtp` vs `pretrain-mlm`: smtp do mask inside `model`, mlm do mask here
    # prepare for predicting 3D coordinate
    input_ids = in_dict["input_ids"] + in_dict["labels"][-1:]  # add eos
    len_extended_tokens = 1
    assert len(gtokenizer.config.get("ensemble_datasets", [])) == 0
    assert gtokenizer.mpe is None
    in_dict["input_ids"] = input_ids
    in_dict["position_ids"].extend(
        list(
            range(
                len(in_dict["position_ids"]),
                len(in_dict["position_ids"]) + len_extended_tokens,
            )
        )
    )
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
    # print("[DEBUG] raw node_idx:\n", node_idx)
    node_mask = get_mask_of_raw_seq(node_idx, mask_type="random")
    # below remove non-mask of element `0`: appearance of `0`
    node_mask = node_mask * (node_idx > 0)
    # print("[DEBUG] node_mask:\n", node_mask)
    edge_seq = list(zip([0] + node_idx.tolist()[:-1], node_idx.tolist()))
    edge_mask = get_mask_of_raw_seq(edge_seq, mask_type="random")
    edge_mask = edge_mask * (np.array(edge_seq) > 0).all(axis=-1)
    # print("[DEBUG] edge_mask:\n", edge_mask)
    node_type = np.vstack([node_idx_clip, node_mask, node_idx, edge_mask]).T
    input_ids = np.hstack([np.array(input_ids), node_type])
    return input_ids


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


@_inputs_deco("graph")
def prepare_inputs_for_graph_lvl_task(
    in_dict: Dict[str, List[Union[int, Iterable[int]]]],
    *,
    graph: Data,
    gtokenizer,
    ls_raw_node_idx: List[int],
    **kwargs,
):
    """
    inputs Graph-level DenoisingDoubleHead tasks:
        task1 -> supervised regression task
        task2 -> denoising task task, e.g., predict 3d coordinates noises
    :param in_dict:
    :param graph:
    :param gtokenizer:
    :param ls_raw_node_idx:
    :param kwargs:
    :return:
    """
    # reserved_semantics_token = gtokenizer.get_common_semantics()[graph.idx_of_ds]
    # token_id = gtokenizer._map_tokens_to_ids(reserved_semantics_token)
    # ls_extend_tokens = [token_id]
    # ABOVE for fine-tuning with train data from two different source, e.g., PCQM4M-v2 & CEPDB
    if (
        gtokenizer.config.get("task_conversion", None)
        == "regression2binary_classification"
    ):
        scale = MOL_ENERGY_SCALE
        max_len = MOL_ENERGY_BIN_LEN
        ls_extend_tokens, bin_labels = _get_bin_inputs_labels(
            in_dict["input_ids"], graph, gtokenizer, scale, max_len
        )
    else:
        ls_extend_tokens = []
        bin_labels = None
    in_dict = _extend_input_dict(in_dict, ls_extend_tokens, cmpe=gtokenizer.cmpe)
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
            # use `default edge-attr` as src-node's edge attr
            assert len(ls_extend_tokens) == 2
            default_edge_attr_id = gtokenizer.get_default_edge_attr_id()
            assert len(default_edge_attr_id) == edge_dim
            ls_extend_tokens[0] = ls_extend_tokens[0][:-edge_dim] + list(
                default_edge_attr_id
            )
            if tgt_edge_attr_token_id:
                # use `tgt_edge_attr` as dst-node's edge attr
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
    in_dict = _extend_input_dict(
        in_dict,
        ls_extend_tokens,
        cmpe=gtokenizer.cmpe,
    )
    in_dict["idx"] = (
        graph.seed_node.tolist() if hasattr(graph, "seed_node") else ls_src_dst
    )
    in_dict["edge_labels"] = graph.y.item()
    if "embed" in in_dict:
        in_dict["embed"].extend(ls_extend_emb)
        assert len(in_dict["input_ids"]) == len(in_dict["embed"])
    if hasattr(graph, "wgt"):
        in_dict["wgt"] = graph.wgt.item()
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
        # for node identity encoding with multiple tokens
        ls_token_ids = list(tgt_node_token_id)
    raw_ls_extend_tokens = list(ls_token_ids)
    ls_extend_tokens = list(ls_token_ids)
    ls_extend_emb = []
    if isinstance(in_dict["input_ids"][0], List):
        dict_mapping = {x[0]: x for x in in_dict["input_ids"]}
        ls_extend_tokens = [list(dict_mapping[x]) for x in raw_ls_extend_tokens]
        edge_dim = gtokenizer.config["semantics"]["edge"]["dim"]
        if edge_dim > 0:
            # use `default edge-attr` as tgt-node's edge attr
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
    in_dict = _extend_input_dict(
        in_dict,
        ls_extend_tokens,
        cmpe=gtokenizer.cmpe,
    )
    in_dict["idx"] = ls_token_ids
    assert graph.num_nodes == graph.y.shape[0]
    in_dict["node_labels"] = graph.y[graph.root_n_id].tolist()
    if "embed" in in_dict:
        in_dict["embed"].extend(ls_extend_emb)
        assert len(in_dict["input_ids"]) == len(in_dict["embed"])
    if hasattr(graph, "wgt"):
        in_dict["wgt"] = graph.wgt.item()
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
    permute_label: bool = True,  # disable when inferring
    **kwargs,
):
    if (
        hasattr(graph, "y")
        and (graph.y is not None)
        and (graph.y.shape[0] == graph.num_nodes)
    ):
        nodev2_labels = graph.y[:, 0].tolist()
    else:  # in inference mode
        nodev2_labels = [-100] * graph.x.shape[0]
    assert len(tgt_node_token_id) == len(nodev2_labels)
    # if permute_label:  # NOT WORKING, model CANNOT be trained!
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
            [token_id] * len(inputs_instance) if isinstance(token_id, int) else token_id
            for token_id in ls_extend_tokens
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


def _get_bin_inputs_labels(
    input_ids, graph, gtokenizer, scale: float = 1000, max_len: int = 16
):
    num_feat = len(input_ids[0]) if isinstance(input_ids[0], list) else 0
    seq = len(input_ids)

    bi_str = bin(int(round(torch.squeeze(graph.y).tolist() * scale)))
    bi_str = bi_str[2:]
    bi_str = "0" * (max_len - len(bi_str)) + bi_str
    bi_ls = list(bi_str)

    token_ids = gtokenizer._map_tokens_to_ids([f"<{ele}>" for ele in bi_ls[:-1]])
    if num_feat > 0:
        ls_extend_tokens = np.array([token_ids] * num_feat).T.tolist()
    else:
        ls_extend_tokens = list(token_ids)
    bin_labels = [-100] * (seq - 1) + [int(ele) for ele in bi_ls]
    return ls_extend_tokens, bin_labels


@dataclass
class TokenizationOutput(ModelOutput):
    """
    Base class for tokenizer's outputs

    Args:
        ls_tokens (`List`):
            List of input tokens.
        ls_labels (`List`):
            List of label tokens.
        ls_raw_node_idx (`List`):
            List of raw node's index.
        tgt_node_token (`str`):
            node => target node token for node-level tasks.
        tgt_edge_src_token (`str`):
            edge => target src node token for edge-level tasks.
        tgt_edge_dst_token (`str`):
            edge => target dst node token for edge-level tasks.
        tgt_edge_attr_token (`str`):
            edge => target edge attr token for edge-level tasks.
        tgt_pos (`int`):
            For UniBi attention mixed model
    """

    ls_tokens: List[Union[str, List[str]]] = None
    ls_labels: List[Union[str, List[str]]] = None
    ls_raw_node_idx: List[int] = None
    tgt_node_token: Union[str, List[str], Tuple[str]] = None
    tgt_edge_src_token: Union[str, List[str], Tuple[str]] = None
    tgt_edge_dst_token: Union[str, List[str], Tuple[str]] = None
    tgt_edge_attr_token: List[str] = None
    tgt_pos: Optional[torch.Tensor] = None
    ls_embed: List[List[float]] = None
    ls_len: List[int] = None


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


DICT_MASK_FUNC = {
    "first": _obtain_first_appearance_idx,
    "last": _obtain_last_appearance_idx,
    "random": _obtain_random_appearance_idx,
    "all": _obtain_all_appearance_idx,
}


def get_mask_of_raw_seq(raw_seq, mask_type="first"):
    deco_seq = [
        (min(ele), max(ele)) if isinstance(ele, tuple) else ele for ele in raw_seq
    ]
    dict_deco_idx = _obtain_all_idx_of_each_element(deco_seq)
    mask_type = (
        random.choice(("first", "last", "random")) if mask_type == "mix" else mask_type
    )
    mask_func = DICT_MASK_FUNC[mask_type]
    idx = mask_func(dict_deco_idx)
    idx = sorted(idx)

    seq_len = len(raw_seq)
    mask = np.zeros(seq_len, dtype=int)
    mask[idx] = 1
    return mask
