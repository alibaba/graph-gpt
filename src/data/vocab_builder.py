import math
import os
import time
import numpy as np
from typing import Dict, List
from pprint import pformat

import torch
import numpy.typing as npt
from torch_geometric.data import Dataset
from .dataset_iterable import OdpsTableIterableTokenizedDataset
from .data_sources import read_merge_molecule_datasets
from ..utils.mol_utils import read_complete_mol_features_ds


def _get_vocab_of_attr(
    attr: npt.ArrayLike, world_identifier: str, node_edge_identifier: str
):
    # node_edge_identifier == "node" or "edge" or "gid"
    _, num_col = attr.shape
    ls_arr = []
    for i in range(num_col):
        col_val = np.sort(np.array(list(set(attr[:, i])))).astype(str).reshape([-1, 1])

        num_row, _ = col_val.shape
        prefix_identifier = np.array(
            [[world_identifier, node_edge_identifier]] * num_row
        )
        col_idx = np.array([[i]] * num_row).astype(str)
        arr = np.hstack([prefix_identifier, col_idx, col_val])
        ls_arr.append(arr)
    return ls_arr


def _get_vocab(ls_arr, ignored_val: str = None):
    if len(ls_arr) > 0:
        semantics_arr = np.vstack(ls_arr)
        raw_vocab = np.unique(semantics_arr, axis=0).astype(str)
        vocab = ["#".join(ele) for ele in raw_vocab if ele[-1] != str(ignored_val)]
        raw_vocab = np.unique(raw_vocab[:, :-1], axis=0)  # remove last col of val
        vocab_default = ["#".join(ele) for ele in raw_vocab]
        vocab = vocab_default + vocab
    else:
        vocab = []
    return vocab


def _get_node_edge_graph_semantics_vocab(dataset: Dataset, config: Dict, neg: str):
    # neg: node_edge_graph
    world_identifier = config.get("attr_world_identifier", config["dataset"])
    assert not (
        config["semantics"][neg]["discrete"] and config["semantics"][neg]["continuous"]
    ), "NotImplement"

    ls_arr = []
    attr = config["semantics"][neg]["discrete"]
    if attr is not None:
        ls_attr_tensor = [g[attr] for g in dataset]
        attr_arr = torch.vstack(ls_attr_tensor).numpy()
        ls_arr += _get_vocab_of_attr(attr_arr, world_identifier, neg)
    print(f"Building {neg} vocab with {len(ls_arr)} discrete {neg} features!")
    discrete_vocab = _get_vocab(ls_arr, config["semantics"][neg]["ignored_val"])

    graph = dataset[0]
    ls_arr = []
    attr = config["semantics"][neg]["continuous"]
    if attr is not None:
        attr_arr = np.ones((1, graph[attr].shape[1]), dtype=np.int64)
        ls_arr += _get_vocab_of_attr(attr_arr, world_identifier, neg)
    print(f"Building {neg} vocab with {len(ls_arr)} continuous {neg} features!")
    continuous_vocab = _get_vocab(ls_arr, None)
    return discrete_vocab + continuous_vocab


def get_semantics_vocab(dataset: Dataset, config: Dict):
    reserved_vocab = config["semantics"]["common"].get("reserved_token", [])
    numbers_vocab = config["semantics"]["common"].get("numbers", [])
    world_identifier = config.get("attr_world_identifier", config["dataset"])
    if world_identifier == "molecule":
        # print(f"merge all molecule datasets to build a unified vocab")
        # new_ds = read_merge_molecule_datasets(config["data_dir"])
        print(
            f"Read artificial molecule dataset that contains all possible features to build a unified vocab"
        )
        new_ds = read_complete_mol_features_ds()
        data = dataset._data
        data.x = new_ds[0].x
        data.edge_attr = new_ds[0].edge_attr
        dataset = [data]
    node_vocab = _get_node_edge_graph_semantics_vocab(dataset, config, "node")
    edge_vocab = _get_node_edge_graph_semantics_vocab(dataset, config, "edge")
    graph_vocab = _get_node_edge_graph_semantics_vocab(dataset, config, "graph")
    return reserved_vocab + numbers_vocab + node_vocab + edge_vocab + graph_vocab


def _get_node_structure_vocab(config: Dict):
    bos_token = config.get("bos_token", "0")
    eos_token = config.get("eos_token", "<eos>")
    s_tokens = [
        config.get("regression_token", None),
        config.get("weight_token", None),
        config.get("summary_token", None),
    ]
    idx_token = [str(ele) for ele in range(1, config["scope_base"])]
    high_lvl_scope = int(math.ceil(config["node_scope"] / config["scope_base"]))
    idx_token_high_lvl = [
        f"{ele}*{config['scope_base']}" for ele in range(1, high_lvl_scope)
    ]
    vocab = s_tokens + [eos_token, bos_token] + idx_token + idx_token_high_lvl
    vocab = [ele for ele in vocab if ele is not None]
    return vocab


def _get_edge_structure_vocab(config: Dict):
    dir_tokens = [
        config.get("in_token", None),
        config.get("out_token", None),
        config.get("bi_token", None),
        config.get("jump_token", None),
    ]
    s_tokens = [
        config.get("regression_token", None),
        config.get("weight_token", None),
        config.get("summary_token", None),
    ]
    vocab = dir_tokens + s_tokens
    vocab = [ele for ele in vocab if ele is not None]
    return vocab


def _get_graph_structure_vocab(config: Dict):
    reg_token = config.get("regression_token", None)
    sum_token = config.get("summary_token", None)
    vocab = [reg_token, sum_token]
    vocab = [ele for ele in vocab if ele is not None]
    return vocab


def _get_common_structure_vocab(config: Dict):
    icl_token = config.get("icl_token", None)
    sep_token = config.get("sep_token", None)
    reserved_token = config.get("reserved_token", [])
    vocab = [icl_token, sep_token] + reserved_token
    vocab = [ele for ele in vocab if ele is not None]
    return vocab


def get_structure_vocab(config: Dict):
    node_vocab = _get_node_structure_vocab(config["node"])
    edge_vocab = _get_edge_structure_vocab(config["edge"])
    graph_vocab = _get_graph_structure_vocab(config["graph"])
    common_vocab = _get_common_structure_vocab(config["common"])
    vocab = common_vocab + graph_vocab + edge_vocab + node_vocab
    return vocab


def save_vocab(vocab: List[str], fn: str):
    vocab_size = len(vocab)
    token_ids = range(1, vocab_size + 1)
    with open(fn, "w+") as fp:
        vocab_mapping = zip(vocab, token_ids)
        to_be_write = [f"{token} {token_id}\n" for token, token_id in vocab_mapping]
        fp.writelines(to_be_write)
    print(f"Finish vocab construction and save it in {fn}!")


def build_vocab(dataset, config, rank=0, use_cache=True):
    if isinstance(dataset, OdpsTableIterableTokenizedDataset):
        print(f"Tokenized dataset loaded, NO need to build vocab!")
        return
    fn = os.path.join(config["name_or_path"], config.get("vocab_file", "vocab"))
    if rank != 0:
        while not os.path.exists(fn):
            print(f"waiting for the vocab to be built by the worker 0!")
            time.sleep(3)
    else:
        if os.path.exists(fn) and use_cache:
            print(f"Vocab is already built and saved in {fn}!")
        else:
            if not os.path.exists(config["name_or_path"]):
                os.makedirs(config["name_or_path"])
            print("\nstart vocab building ...")
            structure_vocab = get_structure_vocab(config["structure"])
            semantics_vocab = get_semantics_vocab(dataset, config)
            vocab = structure_vocab + semantics_vocab
            save_vocab(vocab, fn)


def load_vocab(fn) -> Dict[str, int]:
    print(f"Loading vocab from {fn} ...")
    with open(fn, "r") as fp:
        res = fp.read()
        ls = res.split("\n")
        ls = [ele.strip().split() for ele in ls if len(ele) > 0]
        vocab_map = {k: int(v) for k, v in ls}
    print(f"{pformat(vocab_map, indent=4)}")
    return vocab_map
