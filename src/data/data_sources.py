import os
import random

import torch
from typing import List
from datetime import datetime
from pprint import pformat
from tqdm import tqdm
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.utils import smiles2graph
from src.utils import control_flow, tokenizer_utils, dataset_utils
from .dataset_map import (
    EnsembleNodesEdgesMapDataset,
    ShaDowKHopSeqMapDataset,
    ShaDowKHopSeqFromEdgesMapDataset,
    RandomNodesMapDataset,
    GraphsMapDataset,
    EnsembleGraphsMapDataset,
)

from .dataset_iterable import (
    GraphsIterableDataset,
    OdpsTableIterableDataset,
    OdpsTableIterableTokenizedDataset,
    OdpsIterableDatasetMTP,
    OdpsTableIterableDatasetOneID,
)


_dataset = control_flow.Register()
read_dataset = _dataset.build  # return func results
get_dataset_reader = _dataset.get  # return the func

_molecule = control_flow.Register()


def read_merge_molecule_datasets(data_dir):
    ls_edge_attr = []
    ls_x = []
    for ds in _molecule._register_map.keys():
        print(f"load molecule dataset {ds}!\n")
        _, raw_dataset = read_dataset(ds, data_dir, None, return_valid_test=False)
        ls_edge_attr.append(raw_dataset._data.edge_attr)
        ls_x.append(raw_dataset._data.x)
    data = Data(edge_attr=torch.vstack(ls_edge_attr), x=torch.vstack(ls_x))
    print(f"Merged molecule dataset:\n{data}")
    return [data]


# @_dataset("structure")
def _read_structure(
    data_dir,
    sampling_config,
    *,
    pretrain_mode=False,
    return_valid_test: bool = False,
    with_prob: bool = False,
    **kwargs,
):
    dataset_name = "structure"
    print(f"\n[{datetime.now()}] Loading dataset {dataset_name} from {data_dir} ...")
    dataset = dataset_utils.StructureDataset(root=data_dir)
    print(f"\n[{datetime.now()}] dataset._data -> {dataset._data}")
    # dataset._data -> Data(edge_index=[2, 350555694], num_nodes=81957816)
    # dataset[0] -> Data(edge_index=[2, 598], num_nodes=)
    # data_dir: e.g., "../data/Struct"
    if hasattr(dataset._data, "g"):
        dataset._data.g = dataset._data.g.reshape((-1, 1))
    split_idx = dataset.get_idx_split()
    split_idx_stats = {k: len(v) for k, v in split_idx.items()}
    print(f"[{datetime.now()}]\n{pformat(split_idx_stats)}")
    dataset.idx_split_dict = {
        "train": torch.arange(len(dataset))[:-400000],
        "valid": torch.arange(len(dataset))[-400000:-200000],
    }
    permute_nodes = True
    if return_valid_test:
        # seed = 42
        # # deterministically shuffle based on seed
        # g = torch.Generator()
        # g.manual_seed(seed)
        # indices = torch.randperm(len(dataset), generator=g)
        # train_max_idx = int(len(dataset) * 0.8)
        # valid_max_idx = int(len(dataset) * 0.9)
        #
        # train_idx = indices[:train_max_idx]
        # valid_idx = indices[train_max_idx:valid_max_idx][:100000]
        # test_idx = indices[valid_max_idx:][:100000]
        train_idx = split_idx["train"]
        valid_idx = split_idx["valid"][:100000]
        test_idx = split_idx["valid"][-100000:]

        train_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=train_idx,
            provide_sampler=True,
            with_prob=with_prob,
        )
        valid_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=valid_idx,
            provide_sampler=True,
        )
        test_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=test_idx,
            provide_sampler=True,
        )
        # train/valid/test
        # xx/xx/xx
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            dataset,
        )
    else:
        train_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=torch.arange(len(dataset)),
            permute_nodes=permute_nodes,
            # sample_idx=split_idx["train"],
            provide_sampler=True,
            with_prob=with_prob,
        )
        return train_dataset, dataset


@_dataset("structure")
def _read_random_graph_structure(
    data_dir,
    sampling_config,
    *,
    pretrain_mode=False,
    return_valid_test: bool = False,
    with_prob: bool = False,
    **kwargs,
):
    print(f"\n[{datetime.now()}] Generating dataset randomly ...")
    dataset = GraphsIterableDataset(
        num_nodes_low=10,
        num_nodes_high=101,
        edges_per_node=4,
    )
    return dataset, dataset


@_dataset("oneid")
def _read_oneid(
    data_dir,
    sampling_config,
    *,
    pretrain_mode=False,
    return_valid_test: bool = False,
    **kwargs,
):
    dataset_name = "oneid"
    print(f"\nLoading dataset {dataset_name} ...")
    dataset = dataset_utils.OneIDSmallDataset(root=data_dir)
    print(f"\ndataset._data -> {dataset._data}")
    # dataset._data -> Data(x=[2240695, 1], edge_index=[2, 10174256], edge_attr=[10174256, 2], a2d=[185203, 2])
    # dataset[0] -> Data(x=[143, 1], edge_index=[2, 598], edge_attr=[598, 2], a2d=[4, 2])
    # data_dir: e.g., "../data/OneID"
    split_idx = dataset.get_idx_split()
    if return_valid_test:
        # seed = 42
        # # deterministically shuffle based on seed
        # g = torch.Generator()
        # g.manual_seed(seed)
        # indices = torch.randperm(len(dataset), generator=g)
        # train_max_idx = int(len(dataset) * 0.8)
        # valid_max_idx = int(len(dataset) * 0.9)
        #
        # train_idx = indices[:train_max_idx]
        # valid_idx = indices[train_max_idx:valid_max_idx][:100000]
        # test_idx = indices[valid_max_idx:][:100000]
        train_idx = split_idx["train"]
        valid_idx = split_idx["valid"][:100000]
        test_idx = split_idx["valid"][-100000:]

        train_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=train_idx,
            provide_sampler=True,
        )
        valid_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=valid_idx,
            provide_sampler=True,
        )
        test_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=test_idx,
            provide_sampler=True,
        )
        # train/valid/test
        # xx/xx/xx
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            dataset,
        )
    else:
        train_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=torch.arange(len(dataset)),
            # sample_idx=split_idx["train"],
            provide_sampler=True,
        )
        return train_dataset, dataset


@_dataset("triangles")
def _read_triangles(
    data_dir,
    sampling_config,
    *,
    pretrain_mode=False,
    return_valid_test: bool = False,
    **kwargs,
):
    dataset_name = "TRIANGLES"
    print(f"\nLoading dataset {dataset_name} from {data_dir} ...")
    dataset = TUDataset(root=data_dir, name=dataset_name)
    print(f"\ndataset._data -> {dataset._data}")
    # dataset._data -> Data(x=[938438, 0], edge_index=[2, 2947024], y=[45000])
    # dataset[0] -> Data(edge_index=[2, 56], x=[23, 0], y=[1])
    # max num nodes: 100
    # data_dir: e.g., "../data/TUDataset"
    dataset._data.y = dataset._data.y.to(dtype=torch.int64) - 1
    # dataset._data.x = torch.zeros((dataset._data.x.shape[0], 1), dtype=torch.int64)
    permute_nodes = True
    if return_valid_test:
        indices = torch.arange(len(dataset))

        train_idx = indices[:30000]
        # valid_idx = indices[30000:35000]
        valid_idx = indices[35000:40000]
        # https://github.com/luis-mueller/probing-graph-transformers/blob/main/configs/StructuralAwareness/triangle/triangle-graphormer-t1.yaml
        test_idx = indices[40000:45000]
        # https://github.com/luis-mueller/probing-graph-transformers/blob/main/configs/StructuralAwareness/triangle/triangle-graphormer-t2.yaml
        # max-epochs = 1000; warmup-epochs = 50; base_lr = 1e-3

        train_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=train_idx,
            permute_nodes=permute_nodes,
            provide_sampler=True,
        )
        valid_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=valid_idx,
            permute_nodes=True,
            provide_sampler=True,
        )
        test_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=test_idx,
            permute_nodes=True,
            provide_sampler=True,
        )
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            dataset,
        )
    else:
        train_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=torch.arange(len(dataset)),
            permute_nodes=permute_nodes,
            provide_sampler=True,
        )
        return train_dataset, dataset


@_dataset("reddit_threads")
def _read_reddit_threads(
    data_dir,
    sampling_config,
    *,
    pretrain_mode=False,
    return_valid_test: bool = False,
    **kwargs,
):
    dataset_name = "reddit_threads"
    print(f"\nLoading dataset {dataset_name} from {data_dir} ...")
    dataset = TUDataset(root=data_dir, name=dataset_name)
    print(f"\ndataset._data -> {dataset._data}")
    # pre_transform=tokenizer_utils.add_paths
    # dataset._data -> Data(edge_index=[2, 10094032], y=[203088], num_nodes=4859280)
    # dataset[0] -> Data(edge_index=[2, 20], y=[1], num_nodes=11)
    # data_dir: e.g., "../data/TUDataset"
    if return_valid_test:
        seed = 42
        # deterministically shuffle based on seed
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=g)
        train_max_idx = int(len(dataset) * 0.8)
        valid_max_idx = int(len(dataset) * 0.9)

        train_idx = indices[:train_max_idx]
        valid_idx = indices[train_max_idx:valid_max_idx]
        test_idx = indices[valid_max_idx:]

        train_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=train_idx,
            provide_sampler=True,
        )
        valid_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=valid_idx,
            provide_sampler=True,
        )
        test_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=test_idx,
            provide_sampler=True,
        )
        # train/valid/test
        # 162470/20309/20309
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            dataset,
        )
    else:
        train_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=torch.arange(len(dataset)),
            provide_sampler=True,
        )
        return train_dataset, dataset


@_dataset("molecule")
def _read_molecules(
    data_dir,
    sampling_config,
    *,
    pretrain_mode: bool = False,
    return_valid_test: bool = False,
    with_prob: bool = False,
    true_valid: int = 0,
    ensemble_datasets: List = None,
    **kwargs,
):
    ls_ds = []
    for ds_name in ensemble_datasets:
        dataset, raw_dataset = read_dataset(
            ds_name, data_dir, None, return_valid_test=False
        )
        ls_ds.append(dataset)
    train_dataset = EnsembleGraphsMapDataset(ls_ds)
    return train_dataset, raw_dataset


@_molecule("ogbg-molhiv")
@_dataset("ogbg-molhiv")
def _read_ogbg_molhiv(data_dir, sampling_config, *, pretrain_mode=False, **kwargs):
    dataset_name = "ogbg-molhiv"
    dataset = PygGraphPropPredDataset(root=data_dir, name=dataset_name)
    # dataset._data -> Data(num_nodes=1049163, edge_index=[2, 2259376], edge_attr=[2259376, 3], x=[1049163, 9], y=[41127, 1])
    train_dataset = GraphsMapDataset(
        dataset,
        None,
        sample_idx=torch.arange(len(dataset)),
        provide_sampler=True,
    )
    return train_dataset, dataset


@_molecule("ogbg-molpcba")
@_dataset("ogbg-molpcba")
def _read_ogbg_molpcba(
    data_dir, sampling_config, *, pretrain_mode=False, return_valid_test=False, **kwargs
):
    dataset_name = "ogbg-molpcba"
    print(f"\nLoading dataset {dataset_name} ...")
    dataset = PygGraphPropPredDataset(root=data_dir, name=dataset_name)
    # dataset._data -> Data(num_nodes=11373137, edge_index=[2, 24618372], edge_attr=[24618372, 3], x=[11373137, 9], y=[437929, 128])
    if return_valid_test:
        split_idx = dataset.get_idx_split()
        train_dataset = GraphsMapDataset(
            dataset, None, sample_idx=split_idx["train"], provide_sampler=True
        )
        valid_dataset = GraphsMapDataset(
            dataset, None, sample_idx=split_idx["valid"], provide_sampler=True
        )
        test_dataset = GraphsMapDataset(
            dataset, None, sample_idx=split_idx["test"], provide_sampler=True
        )
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            dataset,
        )
    else:
        train_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=torch.arange(len(dataset)),
            provide_sampler=True,
        )
        return train_dataset, dataset


@_molecule("PCQM4Mv2")
@_dataset("PCQM4Mv2")
def _read_pcqm4mv2(
    data_dir,
    sampling_config,
    *,
    pretrain_mode: bool = False,
    return_valid_test: bool = False,
    with_prob: bool = False,
    true_valid: int = 0,
    **kwargs,
):
    print("\nLoading dataset PCQM4Mv2 ...")
    # data_dir: e.g., "../data/OGB"
    # dataset = dataset_utils.PygPCQM4Mv2PosDataset(root=data_dir)
    # dataset._data -> Data(edge_index=[2, 109093666], edge_attr=[109093666, 3], x=[52970672, 9], y=[3746620], pos=[52970672, 3])
    dataset = PygPCQM4Mv2Dataset(root=data_dir, smiles2graph=smiles2graph)
    # dataset._data -> Data(edge_index=[2, 109093626], edge_attr=[109093626, 3], x=[52970652, 9], y=[3746620])
    # dataset = dataset_utils.PygPCQM4Mv2ExtraDataset(root=data_dir)
    # dataset._data -> Data(edge_index=[2, 109093626], edge_attr=[109093626, 3], x=[52970652, 12], y=[3746620])
    print(f"\ndataset._data -> {dataset._data}")

    if isinstance(dataset, dataset_utils.PygPCQM4Mv2ExtraDataset):
        dataset._data.x = dataset._data.x[:, [0, 4, 5, 9, 10, 11]]
        dataset._data.edge_attr = dataset._data.edge_attr[:, [1]]
        print(f"\ndataset._data -> {dataset._data}")

    # dataset._data.y = dataset._data.y - dataset._data.y[dataset.get_idx_split()["train"]].median()
    # median is the minimum of y if minimizing MAE
    # https://dsc-courses.github.io/dsc40a-2022-fa/resources/lecture/lec02_mahdi.pdf

    permute_nodes = True
    if return_valid_test:
        add_cepdb = False
        add_zinc = False
        ls_idx = obtain_special_molecules(dataset)
        split_idx = dataset.get_idx_split()
        train_idx = remove_special_molecules(split_idx["train"], ls_idx)
        valid_idx = remove_special_molecules(split_idx["valid"], ls_idx)
        if true_valid > 0:
            train_idx, valid_idx, test_idx = add_valid_to_train(
                train_idx, valid_idx, true_valid
            )
        else:
            mid_idx = len(valid_idx) // 2
            test_idx = valid_idx[mid_idx:]
            print(
                f"Using all valid data as valid: {len(valid_idx)}, and last half of valid data as test: {len(test_idx)}!"
            )
        train_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=train_idx,
            permute_nodes=permute_nodes,
            provide_sampler=True,
            with_prob=with_prob,
        )
        if add_cepdb:
            print("\nLoading dataset CEPDB ...")
            # data_dir: e.g., "../data/OGB"
            dataset_cepdb = dataset_utils.PygCEPDBDataset(root=data_dir)
            # "mass", "pce", "voc", "jsc", "e_homo_alpha", "e_gap_alpha", "e_lumo_alpha"
            # "voc", "e_homo_alpha", "e_gap_alpha", "e_lumo_alpha" 's values are of same scale
            # y: [5], y2: [2,5], y4: [2,4,5,6]
            dataset_cepdb._data.y = torch.nan_to_num(
                dataset_cepdb._data.y[:, [5]], nan=0.0
            )
            print(f"\ndataset._data -> {dataset_cepdb._data}")
            cepdb_dataset = GraphsMapDataset(
                dataset_cepdb,
                None,
                sample_idx=torch.arange(len(dataset_cepdb)),
                provide_sampler=True,
                with_prob=with_prob,
            )
            train_dataset = EnsembleGraphsMapDataset([train_dataset, cepdb_dataset])
        if add_zinc:
            print("\nLoading dataset ZINC ...")
            # data_dir: e.g., "../data/OGB"
            dataset_zinc = dataset_utils.PygZINCDataset(root=data_dir)
            # "MWT", "LogP", "Desolv_apolar", "Desolv_polar", "HBD", "HBA", "tPSA", "Charge", "NRB"
            # "Desolv_apolar" values are of same scale as homo-lumo, Desolv_polar is 10 times
            dataset_zinc._data.y = torch.nan_to_num(
                dataset_zinc._data.y[:, [2]], nan=0.0
            )
            print(f"\ndataset._data -> {dataset_zinc._data}")
            zinc_dataset = GraphsMapDataset(
                dataset_zinc,
                None,
                sample_idx=torch.arange(len(dataset_zinc)),
                provide_sampler=True,
                with_prob=with_prob,
            )
            train_dataset = EnsembleGraphsMapDataset([train_dataset, zinc_dataset])

        valid_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=valid_idx,
            provide_sampler=True,
            ensemble_paths=False,
        )
        test_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=test_idx,
            provide_sampler=True,
            ensemble_paths=False,
        )
        # train/valid/test-dev/test-challenge
        # 3378606/73545/147037/147432
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            dataset,
        )
    else:
        ls_idx = obtain_special_molecules(dataset)
        print(f"In pre-train mode, set all valid data's y to nan!")
        valid_idx = dataset.get_idx_split()["valid"]
        y = dataset._data.y.clone()
        y[valid_idx] = float("nan")
        # y[torch.arange(len(dataset))] = float("nan")
        print(f"Before setting, y has {torch.isnan(dataset._data.y).sum()} NANs")
        dataset._data.y = y.reshape([-1, 1]).round(decimals=3)
        print(f"After setting, y has {torch.isnan(dataset._data.y).sum()} NANs")
        try:
            pretrain_idx = _load_idx_from_file(dataset.root, "dedup_idx")
            print(
                f"Using dedup_idx with {len(pretrain_idx)} molecules instead of original {len(dataset)} molecules!"
            )
        except Exception as inst:
            print(inst)
            pretrain_idx = remove_special_molecules(torch.arange(len(dataset)), ls_idx)
        train_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=pretrain_idx,
            permute_nodes=permute_nodes,
            provide_sampler=True,
            with_prob=with_prob,
        )
        return train_dataset, dataset


def _load_idx_from_file(root_dir, fn):
    fn = os.path.join(root_dir, fn)
    print(f"load idx from {fn} ...")
    with open(fn, "r") as fp:
        ls_idx = fp.readlines()
    ls_idx = [int(each.strip()) for each in ls_idx]
    print(f"load idx from {fn} with 1st 10 idx:\n{ls_idx[:10]}")
    return torch.tensor(ls_idx, dtype=torch.int64)


def add_specific_valid_to_train(
    train_idx: torch.Tensor, valid_idx: torch.Tensor, specific_valid_idx: torch.Tensor
):
    valid_idx_for_train = remove_special_molecules(
        valid_idx, specific_valid_idx.tolist()
    )
    seed = 42
    # deterministically shuffle based on seed
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(valid_idx_for_train), generator=g).tolist()
    # use valid samples in training as the test data to check the difference between valid (unseen) and test (seen)
    num_samples = len(specific_valid_idx)
    indices_test = indices[:num_samples]
    # create new index
    new_train_idx = torch.cat([train_idx, valid_idx_for_train], dim=-1)
    new_valid_idx = specific_valid_idx
    new_test_idx = valid_idx_for_train[indices_test].clone()
    print(
        f"ADD lf's specific valid samples into train!!!\ntrain_idx: {len(train_idx)} -> {len(new_train_idx)}\nvalid_idx: {len(valid_idx)} -> {len(new_valid_idx)}\n"
        f"use {len(new_test_idx)} Valid samples in Training as the test data to check the difference between valid (unseen) and test (seen)"
    )
    return new_train_idx, new_valid_idx, new_test_idx


def add_valid_to_train(
    train_idx: torch.Tensor, valid_idx: torch.Tensor, num_remained: int = 5000
):
    seed = 42
    # deterministically shuffle based on seed
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(valid_idx), generator=g)
    indices_train = indices[:-num_remained]
    indices_valid = indices[-num_remained:]
    # use valid samples in training as the test data to check the difference between valid (unseen) and test (seen)
    cnt_test = min(num_remained, len(valid_idx) - num_remained)
    indices_test = indices[:cnt_test]
    extra_train = len(indices_train)  # 200000
    ls_indices_train = [indices_train] * (extra_train // len(indices_train) + 1)
    indices_train = torch.cat(ls_indices_train, dim=-1)[:extra_train]
    new_train_idx = torch.cat([train_idx, valid_idx[indices_train]], dim=-1)
    new_valid_idx = valid_idx[indices_valid].clone()
    new_test_idx = valid_idx[indices_test].clone()
    print(
        f"ADD valid samples into train!!!\ntrain_idx: {len(train_idx)} -> {len(new_train_idx)}\nvalid_idx: {len(valid_idx)} -> {len(new_valid_idx)}\n"
        f"use {len(new_test_idx)} Valid samples in Training as the test data to check the difference between valid (unseen) and test (seen)\n"
        f"top 10 new valid idx:\n{new_valid_idx[:10]}"
    )
    return new_train_idx, new_valid_idx, new_test_idx


def remove_special_molecules(raw_idx: torch.Tensor, removed_idx: List[int]):
    if len(removed_idx) == 0:
        new_idx = raw_idx
    else:
        dtype = raw_idx.dtype
        raw_idx = raw_idx.tolist()
        new_idx = torch.tensor(
            sorted(list(set(raw_idx) - set(removed_idx))), dtype=dtype
        )
    print(
        f"\nRaw indices: {len(raw_idx)}, Removed indices: {len(removed_idx)}, New indices: {len(new_idx)}"
    )
    return new_idx


def obtain_special_molecules(
    dataset,
    *,
    edge0: bool = False,
    node1: bool = False,
    node2: bool = False,
    disconnected: bool = False,
):
    # given a dataset, obtain the index of some kinds of special molecules
    # e.g., molecules with 1 node/ 2 nodes; disconnected molecules
    ls_idx = []
    if edge0:
        ls_idx_edge0 = _obtain_molecules_with_given_attr_val(dataset, "edge_attr", 0)
        ls_idx.extend(ls_idx_edge0)
    if node1:
        ls_idx_node1 = _obtain_molecules_with_given_attr_val(dataset, "x", 1)
        ls_idx.extend(ls_idx_node1)
    if node2:
        ls_idx_node2 = _obtain_molecules_with_given_attr_val(dataset, "x", 2)
        ls_idx.extend(ls_idx_node2)
    if disconnected:
        ls_idx_disconnected = _obtain_disconnected_molecules(dataset)
        ls_idx.extend(ls_idx_disconnected)
    return list(set(ls_idx))


def _obtain_molecules_with_given_attr_val(dataset, attr: str = "x", val: int = 1):
    print(f"\nObtaining the indices of molecules with attr {attr} of cnt val {val}!")
    fn = os.path.join(dataset.root, f"{attr}_{val}")
    if os.path.exists(fn):
        print(f"Load the indices from existing file {fn}")
        with open(fn, "r+") as fp:
            ls = fp.readlines()
            ls_idx = [int(ele.strip()) for ele in ls]
    else:
        print(f"Calculating the indices on the fly ...")
        # ls_idx = [i for i in tqdm(range(len(dataset))) if dataset[i].x.shape[0] == num_nodes]
        # Use below, much faster
        idx = dataset.slices[attr]
        cnt = idx[1:] - idx[:-1]
        ls_idx = (cnt == val).nonzero(as_tuple=True)[0].tolist()
        with open(fn, "w+") as fp:
            to_be_write = [f"{idx}\n" for idx in ls_idx]
            fp.writelines(to_be_write)
        print(f"Finish indices calculating and save it in {fn}!")
    print(
        f"Number of molecules with attr {attr} of cnt val {val} is {len(ls_idx)}, first 10 of them are:\n{ls_idx[:10]}"
    )
    return ls_idx


def _obtain_disconnected_molecules(dataset):
    print(f"\nObtaining the indices of disconnected molecules!")
    fn = os.path.join(dataset.root, f"disconnected")
    if os.path.exists(fn):
        print(f"Load the indices from existing file {fn}")
        with open(fn, "r+") as fp:
            ls = fp.readlines()
            ls_idx = [int(ele.strip()) for ele in ls]
    else:
        print(f"Calculating the indices on the fly ...")
        from torch_geometric.utils import to_networkx
        import networkx as nx

        def _is_connected(dp):
            G = to_networkx(dp, to_undirected="upper").to_undirected()
            return nx.is_connected(G)

        ls_idx = [i for i in tqdm(range(len(dataset))) if not _is_connected(dataset[i])]
        with open(fn, "w+") as fp:
            to_be_write = [f"{idx}\n" for idx in ls_idx]
            fp.writelines(to_be_write)
        print(f"Finish indices calculating and save it in {fn}!")
    print(
        f"Number of disconnected molecules is {len(ls_idx)}, first 10 of them are:\n{ls_idx[:10]}"
    )
    return ls_idx


@_molecule("CEPDB")
@_dataset("CEPDB")
def _read_cepdb(
    data_dir,
    sampling_config,
    *,
    pretrain_mode: bool = False,
    return_valid_test: bool = False,
    with_prob: bool = False,
    true_valid: int = 0,
    **kwargs,
):
    print("\nLoading dataset CEPDB ...")
    # data_dir: e.g., "../data/OGB"
    assert not return_valid_test
    dataset = dataset_utils.PygCEPDBDataset(root=data_dir)
    print(f"\ndataset._data -> {dataset._data}")
    # dataset._data -> Data(edge_index=[2, 154636230], edge_attr=[154636230, 3], x=[64047540, 9], y=[2313028, 1])
    if len(dataset._data.y.shape) >= 2:
        assert dataset._data.y.shape[1] == 7
        # "mass", "pce", "voc", "jsc", "e_homo_alpha", "e_gap_alpha", "e_lumo_alpha"
        enlarge_rate = torch.tensor(
            [[1, 10, 100, 1, 100, 100, 100]], dtype=torch.float32
        )
        dataset._data.y = torch.nan_to_num(
            (dataset._data.y * enlarge_rate).round(decimals=0), nan=0.0
        ).to(torch.int64)
    else:
        dataset._data.y = dataset._data.y.clone().reshape([-1, 1]).round(decimals=3)
    dataset._data.y = None  # tmp for pre-training
    train_dataset = GraphsMapDataset(
        dataset,
        None,
        sample_idx=torch.arange(len(dataset)),
        provide_sampler=True,
        with_prob=with_prob,
    )
    return train_dataset, dataset


@_molecule("ZINC")
@_dataset("ZINC")
def _read_zinc(
    data_dir,
    sampling_config,
    *,
    pretrain_mode: bool = False,
    return_valid_test: bool = False,
    with_prob: bool = False,
    true_valid: int = 0,
    subset: int = 11,
    **kwargs,
):
    print(f"\nLoading dataset ZINC subset {subset} ...")
    # data_dir: e.g., "../data/OGB"
    assert not return_valid_test
    dataset = dataset_utils.PygZINCDataset(root=data_dir, subset=subset)
    print(f"\ndataset._data -> {dataset._data}")
    # dataset._data -> Data(edge_index=[2, 209405292], edge_attr=[209405292, 3], x=[97741772, 9])
    dataset._data.y = None  # tmp for pre-training
    train_dataset = GraphsMapDataset(
        dataset,
        None,
        sample_idx=torch.arange(len(dataset)),
        provide_sampler=True,
        with_prob=with_prob,
    )
    return train_dataset, dataset


@_dataset("ogbn-proteins")
def _read_ogbn_proteins(
    data_dir,
    sampling_config,
    *,
    pretrain_mode=False,
    return_valid_test=False,
    **kwargs,
):
    dataset_name = "ogbn-proteins"
    print(f"Loading {dataset_name} raw data from dir: {data_dir}")
    raw_dataset = PygNodePropPredDataset(root=data_dir, name=dataset_name)
    # Data(num_nodes=132534, edge_index=[2, 79122504], edge_attr=[79122504, 8], node_species=[132534, 1], y=[132534, 112])
    split_idx = raw_dataset.get_idx_split()
    graph = raw_dataset[0]
    graph.id = torch.arange(graph.node_species.shape[0])
    graph.x_gid = _get_global_local_id_from_enumerate(
        graph.node_species, global_id_only=True
    )
    graph.edge_attr = (graph.edge_attr * 1000 - 1).to(dtype=torch.int64)
    # a). Node identity coding with one-token, vocab very large if many num-nodes
    # graph.x = torch.cat([graph.node_species, graph.id.view(-1,1)], dim=1)  # [N, 2]
    # b). Node identity coding with two-tokens, vocab can be small
    graph.x = _get_global_local_id_from_enumerate(
        graph.node_species, global_id_only=False
    )  # [N, 2]
    # graph.x, graph.x_mask = _mask_concat_node_label_as_feat(graph, split_idx["train"])
    # [N, 2+112]  [114,]  add node-label to pretrain/supervise easily cause overfitting

    def _mask_species(subgraph: Data):
        # mask labels of nodes from the same species of the target node
        tgt_species = subgraph.node_species[subgraph.root_n_id].item()
        mask = subgraph.node_species != tgt_species  # [num_nodes, 1]
        mask = mask.expand(subgraph.x.shape).to(torch.int64)  # [num_nodes, num_feat]
        mask[:, :2] = 1
        subgraph.x = subgraph.x * mask
        return subgraph

    print(
        f"\nLoading dataset from {graph} with EnsembleNodesEdgesMapDataset.\nParams:\nsampling_config: {pformat(sampling_config)}"
    )
    save_dir = os.path.join(raw_dataset.root, "metis")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if return_valid_test:
        train_dataset = EnsembleNodesEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=split_idx["train"],
            provide_sampler=True,
            save_dir=save_dir,
        )
        valid_dataset = EnsembleNodesEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=split_idx["valid"],
            provide_sampler=True,
            save_dir=save_dir,
        )
        test_dataset = EnsembleNodesEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=split_idx["test"],
            provide_sampler=True,
            save_dir=save_dir,
        )
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            [graph],
        )  # TODO: use an elegant method to convert back to a dataset obj
    else:
        dataset = EnsembleNodesEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=None,
            provide_sampler=True,
            save_dir=save_dir,
        )
        return dataset, [
            graph
        ]  # TODO: use an elegant method to convert back to a dataset obj


@_dataset("ogbl-ppa")
def _read_ogbl_ppa(
    data_dir,
    sampling_config,
    *,
    pretrain_mode=False,
    return_valid_test=False,
    **kwargs,
):
    dataset_name = "ogbl-ppa"
    print(f"Loading {dataset_name} raw data from dir: {data_dir}")
    raw_dataset = PygLinkPropPredDataset(root=data_dir, name=dataset_name)
    split_edge = raw_dataset.get_edge_split()
    graph = raw_dataset[0]
    # graph -> Data(num_nodes=576289, edge_index=[2, 42463862], x=[576289, 58])
    graph.id = torch.arange(graph.num_nodes)
    graph.x_gid = _get_global_local_id_from_onehot(graph.x, global_id_only=True)
    graph.x = _get_global_local_id_from_onehot(graph.x, global_id_only=False)
    print(
        f"\nLoading dataset from {graph} with ShaDowKHopSeqFromEdgesMapDataset.\nParams:\nsampling_config: {pformat(sampling_config)}"
    )
    if return_valid_test:
        train_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
        )
        valid_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="valid",
        )
        test_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="test",
        )
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            [graph],
        )  # TODO: use an elegant method to convert back to a dataset obj
    else:
        dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
        )
        return dataset, [
            graph
        ]  # TODO: use an elegant method to convert back to a dataset obj


@_dataset("ogbl-ddi")
def _read_ogbl_ddi(
    data_dir,
    sampling_config,
    *,
    pretrain_mode=False,
    return_valid_test=False,
    **kwargs,
):
    dataset_name = "ogbl-ddi"
    print(f"Loading {dataset_name} raw data from dir: {data_dir}")
    raw_dataset = PygLinkPropPredDataset(root=data_dir, name=dataset_name)
    split_edge = raw_dataset.get_edge_split()
    graph = raw_dataset[0]
    # graph -> Data(num_nodes=4267, edge_index=[2, 2135822])
    graph.id = torch.arange(graph.num_nodes)
    graph.x = (torch.arange(graph.num_nodes) + 1).view((-1, 1))
    print(
        f"\nLoading dataset from {graph} with EnsembleNodesEdgesMapDataset.\nParams:\nsampling_config: {pformat(sampling_config)}"
    )
    if return_valid_test:
        train_dataset = EnsembleNodesEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
        )
        valid_dataset = EnsembleNodesEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="valid",
        )
        test_dataset = EnsembleNodesEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="test",
        )
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            [graph],
        )  # TODO: use an elegant method to convert back to a dataset obj
    else:
        dataset = EnsembleNodesEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
        )
        return dataset, [
            graph
        ]  # TODO: use an elegant method to convert back to a dataset obj


@_dataset("odps_oneid")
def _read_odps_oneid_data(table, return_valid_test: bool = False, **kwargs):
    slice_id = int(os.environ.get("RANK", 0))
    slice_count = int(os.environ.get("WORLD_SIZE", 1))

    ls_tables = table.split(",")
    if return_valid_test:
        train_table, valid_table = ls_tables
        train_dataset = OdpsTableIterableDatasetOneID(
            train_table, slice_id, slice_count
        )
        valid_dataset = OdpsTableIterableDatasetOneID(
            valid_table, slice_id, slice_count
        )
        return train_dataset, valid_dataset, valid_dataset, train_dataset
    else:
        train_table = ls_tables[0]
        train_dataset = OdpsTableIterableDatasetOneID(
            train_table, slice_id, slice_count
        )
        return train_dataset, train_dataset


@_dataset("odps")
def _read_odps_data(table, mode="train", **kwargs):
    if mode == "train":
        slice_id = int(os.environ.get("RANK", 0))
        slice_count = int(os.environ.get("WORLD_SIZE", 1))
    else:
        slice_id = 0
        slice_count = 1
    dataset = OdpsTableIterableDataset(table, slice_id, slice_count)
    return dataset, dataset


@_dataset("odps_tokenized")
def _read_odps_tokenized_data(
    table: str, mode: str = "train", supervised_task: str = "", **kwargs
):
    slice_id = int(os.environ.get("RANK", 0))
    slice_count = int(os.environ.get("WORLD_SIZE", 1))

    assert mode in {"train", "all"}
    ls_tables = table.split(",")
    assert (
        len(ls_tables) == 3 if mode == "all" else len(ls_tables) == 1
    ), f"mode: {mode}\nls_tables: {ls_tables}"

    train_table = ls_tables[0]
    train_dataset = OdpsTableIterableTokenizedDataset(
        train_table, slice_id, slice_count, supervised_task, **kwargs
    )
    if mode == "all":
        _, valid_table, test_table = ls_tables
        valid_dataset = OdpsTableIterableTokenizedDataset(
            valid_table, slice_id, slice_count, supervised_task, **kwargs
        )
        test_dataset = OdpsTableIterableTokenizedDataset(
            test_table, slice_id, slice_count, supervised_task, **kwargs
        )
        return train_dataset, valid_dataset, test_dataset, train_dataset
    elif mode == "train":
        return train_dataset, train_dataset
    else:
        raise NotImplementedError(f"Mode {mode} is NOT implemented yet!")


def _get_global_local_id_from_onehot(x: torch.Tensor, global_id_only: bool):
    # x: one-hot tensor
    # assume each col of one-hot tensor `x` represents one species
    # each row-record belongs to one of the species
    global_id = (
        torch.argmax(x, dim=-1, keepdim=True) + 1
    )  # [N, 1] get the col-idx as the global-id, i.e., which species for the record
    if global_id_only:
        output = global_id
    else:
        x_cum = torch.cumsum(
            x, dim=0
        )  # [N, m] cum-sum along each col to get the cumulated count of each species
        idx = (
            x_cum * x
        )  # [N, m] element-wise multiplication to get the local-id of each record
        local_id = idx.sum(dim=-1).view((-1, 1))  # [N, 1]
        output = torch.cat([global_id, local_id], dim=1)  # [N, 2]
    return output


def _get_global_local_id_from_enumerate(x: torch.Tensor, global_id_only: bool):
    # x: tensor of (N,) or (N,1), enumerate all possible elements
    # assume each element represents one species
    # each row-record belongs to one of the species
    global_id = x.view(
        (-1, 1)
    )  # [N, 1] get the col-idx as the global-id, i.e., which species for the record
    if global_id_only:
        output = global_id.clone()
    else:
        ls_enumerates = x.view(-1).tolist()
        dict_ = dict.fromkeys(set(ls_enumerates), 0)
        ls_local_id = []
        for ele in ls_enumerates:
            dict_[ele] = dict_[ele] + 1
            ls_local_id.append(dict_[ele])
        local_id = torch.tensor(ls_local_id, dtype=x.dtype).view((-1, 1))  # [N, 1]
        output = torch.cat([global_id, local_id], dim=1)  # [N, 2]
    return output


def _mask_concat_node_label_as_feat(graph: Data, idx: torch.Tensor):
    assert len(graph.y.shape) == 2
    mask = torch.zeros((graph.num_nodes, 1), dtype=torch.int64)
    mask[idx] = 1
    new_x = torch.cat([graph.x, mask * graph.y], dim=1)  # [num_nodes, num_feat]
    feat_mask = torch.cat(
        [
            torch.ones(graph.x.shape[1], dtype=torch.int64),
            torch.zeros(graph.y.shape[1], dtype=torch.int64),
        ]
    )  # [num_feat]
    return new_x, feat_mask
