import os
import time

import torch
from typing import List
from datetime import datetime
from pprint import pformat
import numpy as np

from tqdm import tqdm
import torch.distributed as dist
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.utils.undirected import is_undirected
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.utils import smiles2graph
from src.utils import control_flow, dataset_utils, mol_utils
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


@_dataset("spice-circuit")
def _read_spice_circuit(
    data_dir,
    sampling_config,
    *,
    pretrain_mode=False,
    return_valid_test: bool = False,
    **kwargs,
):
    dataset_name = "spice-circuit"
    print(f"\nLoading dataset {dataset_name} ...")
    dataset = dataset_utils.SpiceCircuitDataset(root=data_dir)
    print(f"\ndataset._data -> {dataset._data}")
    # dataset._data -> Data(x=[348319, 1], edge_index=[2, 3014676], y=[3350])
    # dataset[0] -> Data(x=[8, 1], edge_index=[2, 18], y=[1])
    # data_dir: e.g., "../data/Custom"
    split_idx = dataset.get_idx_split()
    if return_valid_test:
        train_idx = split_idx["train"]
        valid_idx = split_idx["valid"]
        test_idx = split_idx["test"]

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


@_dataset("structure")
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
    data_ver = "v05"
    print(
        f"\n[{datetime.now()}] Loading dataset {dataset_name} of {data_ver} from {data_dir} ..."
    )
    dataset = dataset_utils.StructureDataset(root=data_dir, data_ver=data_ver)
    print(f"\n[{datetime.now()}] dataset._data -> {dataset._data}")
    # dataset._data -> Data(edge_index=[2, 350555694], num_nodes=81957816)
    # dataset[0] -> Data(edge_index=[2, 598], num_nodes=)
    # data_dir: e.g., "../data/TUDataset"
    if hasattr(dataset._data, "g"):
        dataset._data.g = dataset._data.g.reshape((-1, 1))
    split_idx = dataset.get_idx_split()
    split_idx_stats = {k: len(v) for k, v in split_idx.items()}
    print(f"[{datetime.now()}]\n{pformat(split_idx_stats)}")
    # reset `get_idx_split()` results, mainly to be used in `train_pretrain.py`
    dataset.idx_split_dict = {
        "train": torch.arange(len(dataset))[:-400000],
        "valid": torch.arange(len(dataset))[-400000:-200000],
    }
    permute_nodes = True
    remove_reddit = True
    remove_tri = True
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
        if remove_reddit:
            split_idx = {k: v for k, v in split_idx.items() if "reddit" not in str(k)}
        if remove_tri:
            split_idx = {
                k: v for k, v in split_idx.items() if "triangles" not in str(k)
            }
        ls_idx = [v for k, v in split_idx.items()]
        idx = torch.unique(torch.cat(ls_idx))
        print(
            f"remove_reddit={remove_reddit}, remove_tri={remove_tri} => {len(dataset)} -> {len(idx)}"
        )
        train_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=idx,
            permute_nodes=permute_nodes,
            # sample_idx=split_idx["train"],
            provide_sampler=True,
            with_prob=with_prob,
        )
        return train_dataset, dataset


# @_dataset("structure")
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
        # # A). deterministically shuffle based on seed
        # seed = 42
        # g = torch.Generator()
        # g.manual_seed(seed)
        # indices = torch.randperm(len(dataset), generator=g)
        # train_max_idx = int(len(dataset) * 0.8)
        # valid_max_idx = int(len(dataset) * 0.9)
        #
        # train_idx = indices[:train_max_idx]
        # valid_idx = indices[train_max_idx:valid_max_idx]
        # test_idx = indices[valid_max_idx:]

        # B). random shuffle to run multiple exps to average the results
        g = torch.Generator()
        indices = torch.randperm(len(dataset), generator=g)
        train_max_idx = int(len(dataset) * 0.8)
        train_idx = indices[:train_max_idx]
        valid_idx = indices[train_max_idx:]
        test_idx = indices[-8:]

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
            ds_name, data_dir, None, return_valid_test=False, pt_all=True
        )
        ls_ds.append(dataset)
    train_dataset = EnsembleGraphsMapDataset(ls_ds)
    return train_dataset, raw_dataset


@_molecule("ogbg-molhiv")
@_dataset("ogbg-molhiv")
def _read_ogbg_molhiv(data_dir, sampling_config, *, return_valid_test=False, **kwargs):
    dataset_name = "ogbg-molhiv"
    dataset = PygGraphPropPredDataset(root=data_dir, name=dataset_name)
    # dataset._data -> Data(num_nodes=1049163, edge_index=[2, 2259376], edge_attr=[2259376, 3], x=[1049163, 9], y=[41127, 1])
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
        # Train: 32901, Valid: 4113, Test: 4113
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
            permute_nodes=True,
        )
        return train_dataset, dataset


@_molecule("ogbg-molpcba")
@_dataset("ogbg-molpcba")
def _read_ogbg_molpcba(data_dir, sampling_config, *, return_valid_test=False, **kwargs):
    dataset_name = "ogbg-molpcba"
    print(f"\nLoading dataset {dataset_name} ...")
    dataset = PygGraphPropPredDataset(root=data_dir, name=dataset_name)
    # dataset._data -> Data(num_nodes=11373137, edge_index=[2, 24618372], edge_attr=[24618372, 3], x=[11373137, 9], y=[437929, 128])
    # 437,929 molecules
    split_idx = dataset.get_idx_split()
    if return_valid_test:
        train_dataset = GraphsMapDataset(
            dataset, None, sample_idx=split_idx["train"], provide_sampler=True
        )
        valid_dataset = GraphsMapDataset(
            dataset, None, sample_idx=split_idx["valid"], provide_sampler=True
        )
        test_dataset = GraphsMapDataset(
            dataset, None, sample_idx=split_idx["test"], provide_sampler=True
        )
        # train/valid/test
        # 350343/43793/43793
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
            sample_idx=split_idx["train"],
            provide_sampler=True,
            permute_nodes=True,
        )
        return train_dataset, dataset


@_molecule("PCQM4Mv2")
@_dataset("PCQM4Mv2")
def _read_pcqm4mv2(
    data_dir,
    sampling_config,
    *,
    return_valid_test: bool = False,
    with_prob: bool = False,
    true_valid: int = -1,
    pt_all: bool = False,  # whether to use all data in pre-train
    **kwargs,
):
    print("\nLoading dataset PCQM4Mv2 ...")
    # data_dir: e.g., "../data/OGB"
    # CC means `chiral-center` -> dataset_utils.py::mol2graph_cc
    # dataset = dataset_utils.PygPCQM4Mv2RdkitPosCCDataset(root=data_dir)
    # dataset._data -> Data(edge_index=[2, 109093626], edge_attr=[109093626, 3], x=[52970652, 9], y=[3746620], pos=[52970652, 3], rdkit_pos=[52970652, 3])
    # dataset = dataset_utils.PygPCQM4Mv2PosCCDataset(root=data_dir)  # For `use_3D`
    # dataset._data -> Data(edge_index=[2, 109093666], edge_attr=[109093666, 3], x=[52970672, 9], y=[3746620], pos=[52970672, 3]) -> geometric_data_processed_3d_cc.pt
    # dataset = dataset_utils.PygPCQM4Mv2PosDataset(root=data_dir)
    # dataset._data -> Data(edge_index=[2, 109093666], edge_attr=[109093666, 3], x=[52970672, 9], y=[3746620], pos=[52970672, 3]) -> geometric_data_processed_3d.pt
    # dataset._data -> Data(edge_index=[2, 109093626], edge_attr=[109093626, 3], x=[52970652, 9], y=[3746620], pos=[52970652, 3]) -> geometric_data_processed_3dm_v2.pt
    dataset = PygPCQM4Mv2Dataset(
        root=data_dir, smiles2graph=smiles2graph
    )  # official version
    # dataset._data -> Data(edge_index=[2, 109093626], edge_attr=[109093626, 3], x=[52970652, 9], y=[3746620])
    # dataset = dataset_utils.PygPCQM4Mv2ExtraDataset(root=data_dir)
    # dataset._data -> Data(edge_index=[2, 109093626], edge_attr=[109093626, 3], x=[52970652, 12], y=[3746620])
    print(f"\ndataset._data -> {dataset._data}")

    if isinstance(dataset, dataset_utils.PygPCQM4Mv2ExtraDataset):
        dataset._data.x = dataset._data.x[:, [0, 4, 5, 9, 10, 11]]
        dataset._data.edge_attr = dataset._data.edge_attr[:, [1]]
        print(f"\ndataset._data -> {dataset._data}")

    # BELOW 3 lines are for testing generation capability
    # dataset._data.x = dataset._data.x[:, 0:1]
    # dataset._data.edge_attr = dataset._data.edge_attr[:, 0:1]
    # print(f"\ndataset._data -> {dataset._data}")

    # dataset._data.y = dataset._data.y - dataset._data.y[dataset.get_idx_split()["train"]].median()
    # median is the minimum of y if minimizing MAE
    # https://dsc-courses.github.io/dsc40a-2022-fa/resources/lecture/lec02_mahdi.pdf
    dict_bounds = None
    # if hasattr(dataset._data, "pos") and dataset._data.pos is not None:
    #     dict_bounds = _load_pos_percentile_boundaries(dataset)
    #     for val in dict_bounds.values():  # set lower/upper bounds of boundaries
    #         val[0] = -100
    #         val[-1] = 100
    #     print(dict_bounds)

    permute_nodes = True
    split_idx = dataset.get_idx_split()
    if return_valid_test:
        shift_distribution = False
        add_cepdb = False
        add_zinc = False
        ls_idx = obtain_special_molecules(dataset)
        train_idx = remove_special_molecules(split_idx["train"], ls_idx)
        valid_idx = remove_special_molecules(split_idx["valid"], ls_idx)
        assert true_valid >= -2, f"true_valid: {true_valid}"
        if true_valid == 0:
            old_train, old_valid = train_idx, valid_idx
            train_idx = torch.cat([train_idx, valid_idx], dim=-1)
            valid_len = len(valid_idx)
            test_idx = valid_idx[-valid_len // 16 :]
            valid_idx = valid_idx[-valid_len // 8 :]
            print(
                f"ADD all valid samples into train!!!\n"
                f"train_idx: {len(old_train)} -> {len(train_idx)}\n"
                f"valid_idx: {len(old_valid)} -> {len(valid_idx)}, 1/8 of original valid\n"
                f"test_idx: 0 -> {len(test_idx)}, 1/16 of valid index\n"
                f"Caution: valid/test performance cannot reflect model performance because they are all used in training"
            )
        elif true_valid > 0:
            train_idx, valid_idx, _ = add_valid_to_train(
                train_idx, valid_idx, true_valid
            )
            test_idx = get_large_mols_as_test_from_valid(valid_idx, dataset)
            if len(valid_idx) > 1000:
                test_idx = duplicate_sample_idx(test_idx, 10)
        elif true_valid == -1:
            test_idx = get_large_mols_as_test_from_valid(valid_idx, dataset)
        else:
            _, valid_idx, _ = add_valid_to_train(train_idx, valid_idx, 10000)
            test_idx = split_idx["test-dev"]
            print(
                f"Using all valid data as valid: {len(valid_idx)}, and test-dev data as test: {len(test_idx)}!"
            )
        train_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=train_idx,
            permute_nodes=permute_nodes,
            provide_sampler=True,
            with_prob=with_prob,
            shift_distribution=shift_distribution,
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
        train_dataset.dict_bounds = dict_bounds
        valid_dataset.dict_bounds = dict_bounds
        test_dataset.dict_bounds = dict_bounds
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            dataset,
        )
    else:
        ls_idx = obtain_special_molecules(dataset)
        try:
            fn = "dedup_idx" if pt_all else "dedup_idx_train_valid"
            while not os.path.exists(os.path.join(dataset.root, fn)):
                if int(dist.get_rank()) == 0:
                    _generate_dedup_idx(dataset, fn, pt_all)
                else:
                    seconds = 3
                    print(f"sleep for {seconds} seconds ...")
                    time.sleep(seconds)
            pretrain_idx = _load_idx_from_file(dataset.root, fn)
            print(
                f"Using dedup_idx with {len(pretrain_idx)} molecules instead of original {len(dataset)} molecules!"
            )
        except Exception as inst:
            print(inst)
            pretrain_idx = remove_special_molecules(torch.arange(len(dataset)), ls_idx)
        if not pt_all:
            non_pretrain_mols = torch.cat(
                [split_idx["test-dev"], split_idx["test-challenge"]]
            ).tolist()
            pretrain_idx = remove_special_molecules(pretrain_idx, non_pretrain_mols)
        # pretrain_idx = split_idx["train"]  # ONLY for PosPredict pre-train
        print(f"Use only {len(pretrain_idx)} samples for training!!!")
        train_dataset = GraphsMapDataset(
            dataset,
            None,
            sample_idx=pretrain_idx,
            permute_nodes=permute_nodes,
            provide_sampler=True,
            with_prob=with_prob,
        )
        train_dataset.dict_bounds = dict_bounds
        return train_dataset, dataset


def _load_rotated_pos(dataset):
    fn = "rotate_v3_pos.pt"
    full_fn = os.path.join(dataset.root, fn)
    while not os.path.exists(full_fn):
        if int(os.environ.get("RANK", 0)) == 0:
            samples = dataset.get_idx_split()["train"].tolist()
            ls_pos = [
                mol_utils.rotate_3d_v3(dataset[idx].pos)
                for idx in tqdm(samples)
                if (~(dataset[idx].pos.abs() < 1e-8)).all()
            ]
            sampled_pos = torch.cat(ls_pos)
            torch.save(sampled_pos, full_fn)
        else:
            seconds = 3
            print(f"sleep for {seconds} seconds ...")
            time.sleep(seconds)
    pos = torch.load(full_fn)
    return pos


def _load_pos_percentile_boundaries(dataset):
    num_boundaries = [128, 256, 512, 1024]
    eps = 1e-4
    ls_fn = [f"pos_{num}percentile_eps{eps}_boundaries.pt" for num in num_boundaries]
    ls_full_fn = [os.path.join(dataset.root, fn) for fn in ls_fn]
    ls_boundaries = []
    for num_bins, full_fn in zip(num_boundaries, ls_full_fn):
        while not os.path.exists(full_fn):
            sampled_pos = _load_rotated_pos(dataset)
            if int(os.environ.get("RANK", 0)) == 0:
                np_pos = sampled_pos.numpy()
                filtered_pos = np_pos[abs(np_pos) > eps]
                q = 100 * np.arange(num_bins + 1) / num_bins
                pos_percentile = np.percentile(filtered_pos, q)
                torch.save(torch.tensor(pos_percentile), full_fn)
            else:
                seconds = 3
                print(f"sleep for {seconds} seconds ...")
                time.sleep(seconds)
        print(f"loading boundaries from {full_fn} ...")
        boundaries = torch.load(full_fn)
        print(f"boundaries for {num_bins} is:\n{boundaries}")
        ls_boundaries.append(boundaries)
    return dict(zip(num_boundaries, ls_boundaries))


def _load_idx_from_file(root_dir, fn):
    fn = os.path.join(root_dir, fn)
    print(f"load idx from {fn} ...")
    with open(fn, "r") as fp:
        ls_idx = fp.readlines()
    ls_idx = [int(each.strip()) for each in ls_idx]
    print(f"load idx from {fn} with 1st 10 idx:\n{ls_idx[:10]}")
    return torch.tensor(ls_idx, dtype=torch.int64)


def _generate_dedup_idx(dataset, fn, pt_all=False):
    import pandas as pd

    raw_dir = dataset.raw_dir
    print(f"loading raw data from {os.path.join(raw_dir, 'data.csv.gz')}!")
    data_df = pd.read_csv(os.path.join(raw_dir, "data.csv.gz"))
    split_idx = dataset.get_idx_split()
    if pt_all:
        idx = torch.arange(len(dataset)).tolist()
    else:
        idx = torch.cat([split_idx["train"], split_idx["valid"]]).tolist()
    data_df = data_df.iloc[idx]
    df_new = data_df.drop_duplicates(subset=["smiles"])
    ls_idx = df_new["idx"].values.tolist()
    fn_out = os.path.join(dataset.root, fn)
    print(
        f"Writing {len(ls_idx)} deduped index to {fn_out} out of {len(data_df)} allowable index"
    )
    with open(fn_out, "w") as fp:
        fp.writelines([f"{x}\n" for x in ls_idx])


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
        # f"use {len(new_test_idx)} Valid samples in Training as the test data to check the difference between valid (unseen) and test (seen)\n"
        f"top 10 new valid idx:\n{new_valid_idx[:10]}"
    )
    return new_train_idx, new_valid_idx, new_test_idx


def get_large_mols_as_test_from_valid(valid_idx: torch.Tensor, dataset):
    threshold = 18
    test_idx = [idx for idx in valid_idx.tolist() if dataset[idx].num_nodes > threshold]
    test_idx = torch.tensor(test_idx, dtype=torch.int64)
    print(
        f"use {len(test_idx)} large mols from valid data with num_nodes > {threshold} as the test data\n"
        f"top 10 new test idx:\n{test_idx[:10]}"
    )
    return test_idx


def duplicate_sample_idx(idx: torch.Tensor, rate: int = 10):
    len_ = len(idx)
    idx = idx.tolist() * rate
    idx = torch.tensor(idx, dtype=torch.int64)
    print(f"Duplicate idx {rate} times: {len_} -> {len(idx)}")
    return idx


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


@_dataset("ogbn-products")
def _read_ogbn_products(
    data_dir,
    sampling_config,
    *,
    pretrain_mode=False,
    return_valid_test=False,
    true_valid: int = -1,
    **kwargs,
):
    dataset_name = "ogbn-products"
    print(f"Loading {dataset_name} raw data from dir: {data_dir}")
    raw_dataset = PygNodePropPredDataset(root=data_dir, name=dataset_name)
    split_idx = raw_dataset.get_idx_split()
    graph = raw_dataset[0]
    # graph -> Data(num_nodes=2449029, edge_index=[2, 123718280], x=[2449029, 100], y=[2449029, 1])
    edge_index, _ = remove_self_cycle(graph.edge_index, None)
    assert is_undirected(
        edge_index
    ), f"Graph is NOT undirected, please convert it to undirected using our customized `to_undirected` in this script, AND add one more dim to edge-attr!"
    dividend = 25000
    graph = Data(
        num_nodes=graph.num_nodes,
        edge_index=edge_index,
        x=_get_global_local_id_from_num_nodes(
            graph.num_nodes, dividend=dividend, global_id_only=False
        ),
        x_embed=graph.x,
        id=torch.arange(graph.num_nodes),
        y=graph.y,
    )
    print(
        f"\nLoading dataset from {graph} with ShaDowKHopSeqMapDataset.\nParams:\nsampling_config: {pformat(sampling_config)}"
    )
    if return_valid_test:
        train_dataset = ShaDowKHopSeqMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=split_idx["train"],
            provide_sampler=True,
        )

        print(f"[{datetime.now()}] Reformatting valid data ...")
        valid_dataset = ShaDowKHopSeqMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=split_idx["valid"],
            provide_sampler=True,
        )

        print(f"[{datetime.now()}] Reformatting test data ...")
        test_idx = split_idx["test"]
        if true_valid != -2:
            cnt = test_idx.shape[0]
            ratio = 0.1
            seed = 42
            g = torch.Generator()
            g.manual_seed(seed)
            indices = torch.randperm(cnt, generator=g)
            indices = indices[: int(cnt * ratio)]
            test_idx = test_idx[indices]
            print(
                f"Random sampling {ratio*100} percent test samples: {test_idx.shape[0]}"
            )
        test_dataset = ShaDowKHopSeqMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=test_idx,
            provide_sampler=True,
        )
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            [graph],
        )
    else:
        dataset = ShaDowKHopSeqMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=None,
            provide_sampler=True,
        )
        return dataset, [graph]


@_dataset("ogbn-arxiv")
def _read_ogbn_arxiv(
    data_dir,
    sampling_config,
    *,
    pretrain_mode=False,
    return_valid_test=False,
    true_valid: int = -1,
    **kwargs,
):
    dataset_name = "ogbn-arxiv"
    print(f"Loading {dataset_name} raw data from dir: {data_dir}")
    raw_dataset = PygNodePropPredDataset(root=data_dir, name=dataset_name)
    split_idx = raw_dataset.get_idx_split()
    graph = raw_dataset[0]
    # graph -> Data(num_nodes=169343, edge_index=[2, 1166243], x=[169343, 128], node_year=[169343, 1], y=[169343, 1])
    edge_index, _ = remove_self_cycle(graph.edge_index, None)
    edge_index, edge_attr = to_undirected(edge_index, None)
    dividend = 25000
    graph = Data(
        num_nodes=graph.num_nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        x=_get_global_local_id_from_enumerate_with_dividend(
            graph.node_year.view(-1) - graph.node_year.min(),
            dividend=dividend,
            global_id_only=False,
        ),
        x_embed=graph.x,
        id=torch.arange(graph.num_nodes),
        y=graph.y,
    )
    print(
        f"\nLoading dataset from {graph} with ShaDowKHopSeqMapDataset.\nParams:\nsampling_config: {pformat(sampling_config)}"
    )
    if return_valid_test:
        train_dataset = ShaDowKHopSeqMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=split_idx["train"],
            provide_sampler=True,
        )

        valid_dataset = ShaDowKHopSeqMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=split_idx["valid"],
            provide_sampler=True,
        )

        test_dataset = ShaDowKHopSeqMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=split_idx["test"],
            provide_sampler=True,
        )
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            [graph],
        )
    else:
        dataset = ShaDowKHopSeqMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=None,
            provide_sampler=True,
        )
        return dataset, [graph]


@_dataset("ogbn-papers100M")
def _read_ogbn_papers100M(
    data_dir,
    sampling_config,
    *,
    pretrain_mode=False,
    return_valid_test=False,
    true_valid: int = -1,
    **kwargs,
):
    # cost TOO MUCH memory in `_pre_transform_ogbn_papers100M`, NOT applicable @ 2025-01
    dataset_name = "ogbn-papers100M"
    print(f"Loading {dataset_name} raw data from dir: {data_dir}")
    raw_dataset = PygNodePropPredDataset(
        root=data_dir, name=dataset_name, pre_transform=_pre_transform_ogbn_papers100M
    )
    split_idx = raw_dataset.get_idx_split()
    graph = raw_dataset[0]
    # graph -> Data(num_nodes=111059956, edge_index=[2, 1615685872], x=[111059956, 128], node_year=[111059956, 1], y=[111059956, 1])
    print(
        f"\nLoading dataset from {graph} with ShaDowKHopSeqMapDataset.\nParams:\nsampling_config: {pformat(sampling_config)}"
    )
    if return_valid_test:
        train_dataset = ShaDowKHopSeqMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=split_idx["train"],
            provide_sampler=True,
        )

        valid_dataset = ShaDowKHopSeqMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=split_idx["valid"],
            provide_sampler=True,
        )

        test_dataset = ShaDowKHopSeqMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=split_idx["test"],
            provide_sampler=True,
        )
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            [graph],
        )
    else:
        dataset = ShaDowKHopSeqMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=None,
            provide_sampler=True,
        )
        return dataset, [graph]


def _pre_transform_ogbn_papers100M(graph):
    # graph.node_year.min() -> -1, next min -> 1800
    print(f"applying _pre_transform_ogbn_papers100M ...")
    edge_index, _ = remove_self_cycle(graph.edge_index, None)
    edge_index, edge_attr = to_undirected(edge_index, None)
    dividend = 25000
    node_year = torch.clip(graph.node_year, min=1799)
    graph = Data(
        num_nodes=graph.num_nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        x=_get_global_local_id_from_enumerate_with_dividend(
            node_year.view(-1) - node_year.min(),
            dividend=dividend,
            global_id_only=False,
        ),
        x_embed=graph.x,
        id=torch.arange(graph.num_nodes),
        y=graph.y,
    )
    return graph


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
        )
    else:
        dataset = EnsembleNodesEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            sample_idx=None,
            provide_sampler=True,
            save_dir=save_dir,
        )
        return dataset, [graph]


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
        )
    else:
        dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
        )
        return dataset, [graph]


@_dataset("ogbl-citation2")
def _read_ogbl_citation2(
    data_dir,
    sampling_config,
    *,
    pretrain_mode=False,
    return_valid_test=False,
    true_valid: int = -1,
    **kwargs,
):
    dataset_name = "ogbl-citation2"
    print(f"Loading {dataset_name} raw data from dir: {data_dir}")
    raw_dataset = PygLinkPropPredDataset(root=data_dir, name=dataset_name)
    split_edge = raw_dataset.get_edge_split()
    graph = raw_dataset[0]
    # graph -> Data(num_nodes=2927963, edge_index=[2, 30387995], x=[2927963, 128], node_year=[2927963, 1])
    edge_index, _ = remove_self_cycle(graph.edge_index, None)
    edge_index, edge_attr = to_undirected(edge_index, None)
    dividend = 25000
    graph = Data(
        num_nodes=graph.num_nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        x=_get_global_local_id_from_enumerate_with_dividend(
            graph.node_year.view(-1) - 1901, dividend=dividend, global_id_only=False
        ),
        x_embed=graph.x,
        id=torch.arange(graph.num_nodes),
    )
    print(
        f"\nLoading dataset from {graph} with ShaDowKHopSeqFromEdgesMapDataset.\nParams:\nsampling_config: {pformat(sampling_config)}"
    )
    split_edge["train"].update({"edge": graph.edge_index.T.clone()})
    unique_node_in_edge_idx = torch.unique(edge_index)
    allow_zero_edges = False
    if len(unique_node_in_edge_idx) < graph.num_nodes:
        allow_zero_edges = True
        print(
            f"unique-node-in-edge-index < num_nodes: {len(unique_node_in_edge_idx)} < {graph.num_nodes}!!!\n"
            f"isolated nodes exists, two isolated nodes will form a zero-edge subgraph!!!\n"
            f"set `allow_zero_edges` to be {allow_zero_edges}"
        )
    if return_valid_test:
        cnt_valid_test = 800
        print(f"[{datetime.now()}] Reformatting train data ...")
        edge = torch.vstack(
            [split_edge["train"]["source_node"], split_edge["train"]["target_node"]]
        ).T.clone()
        pos_edge_attr = torch.ones((edge.shape[0], 1), dtype=torch.int64)
        neg_edge_attr_candidates = torch.ones((1, 1), dtype=torch.int64)
        # [N_p,2],  [N_p,1],  [1,1]
        split_edge["train"].update(
            {
                "edge": edge,
                "pos_edge_attr": pos_edge_attr,
                "neg_edge_attr_candidates": neg_edge_attr_candidates,
            }
        )
        train_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
            allow_zero_edges=allow_zero_edges,
        )

        print(f"[{datetime.now()}] Reformatting valid data ...")
        scope = split_edge["valid"]["source_node"].shape[0]
        if true_valid == -2:
            cnt_valid_test = scope
        idx_sel = _get_fixed_sampled_sorted_idx(scope, cnt_valid_test)
        dict_new = _get_reformatted_data_of_citation2(split_edge["valid"], idx_sel)
        split_edge["valid"].update(dict_new)
        valid_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="valid",
            allow_zero_edges=allow_zero_edges,
        )

        print(f"[{datetime.now()}] Reformatting test data ...")
        scope = split_edge["test"]["source_node"].shape[0]
        if true_valid == -2:
            cnt_valid_test = scope
        idx_sel = _get_fixed_sampled_sorted_idx(scope, cnt_valid_test)
        dict_new = _get_reformatted_data_of_citation2(split_edge["test"], idx_sel)
        split_edge["test"].update(dict_new)
        test_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="test",
            allow_zero_edges=allow_zero_edges,
        )
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            [graph],
        )
    else:
        dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
            allow_zero_edges=allow_zero_edges,
        )
        return dataset, [graph]


@_dataset("ogbl-wikikg2")
def _read_ogbl_wikikg2(
    data_dir,
    sampling_config,
    *,
    pretrain_mode=False,
    return_valid_test=False,
    **kwargs,
):
    dataset_name = "ogbl-wikikg2"
    print(f"Loading {dataset_name} raw data from dir: {data_dir}")
    raw_dataset = PygLinkPropPredDataset(root=data_dir, name=dataset_name)
    split_edge = raw_dataset.get_edge_split()
    graph = raw_dataset[0]
    # graph -> Data(num_nodes=2500604, edge_index=[2, 16109182], edge_reltype=[16109182, 1])
    edge_index, edge_attr = remove_self_cycle(graph.edge_index, graph.edge_reltype)
    edge_index, edge_attr = to_undirected(edge_index, edge_attr)
    print(
        f"[WARNING] reltype `95` & `145` are lost when removing duplicated edge-index, please add them back to vocab: 'ogbl-wikikg2#edge#1#95' & 'ogbl-wikikg2#edge#1#145'\n"
        * 5
    )
    dividend = 25000
    graph = Data(
        num_nodes=graph.num_nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        x=_get_global_local_id_from_num_nodes(
            graph.num_nodes, dividend=dividend, global_id_only=False
        ),
        id=torch.arange(graph.num_nodes),
    )
    print(
        f"\nLoading dataset from {graph} with ShaDowKHopSeqFromEdgesMapDataset.\nParams:\nsampling_config: {pformat(sampling_config)}"
    )
    split_edge["train"].update({"edge": graph.edge_index.T.clone()})
    if return_valid_test:
        cnt_valid_test = 800
        print(f"[{datetime.now()}] Reformatting train data ...")
        edge = torch.vstack(
            [split_edge["train"]["head"], split_edge["train"]["tail"]]
        ).T.clone()
        rel = split_edge["train"]["relation"]
        pos_edge_attr = torch.vstack([torch.ones_like(rel), rel]).T.clone()
        unique_rel = torch.unique(rel)
        neg_edge_attr_candidates = torch.vstack(
            [torch.ones_like(unique_rel), unique_rel]
        ).T.clone()
        # edge -> [N_p,2],  pos_edge_attr -> [N_p,2],  neg_edge_attr_candidates -> [C,2]
        # edge -> [16109182, 2],  pos_edge_attr -> [16109182, 2],  neg_edge_attr_candidates -> [535, 2]
        split_edge["train"].update(
            {
                "edge": edge,
                "pos_edge_attr": pos_edge_attr,
                "neg_edge_attr_candidates": neg_edge_attr_candidates,
            }
        )
        train_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
        )

        print(f"[{datetime.now()}] Reformatting valid data ...")
        scope = split_edge["valid"]["head"].shape[0]
        idx_sel = _get_fixed_sampled_sorted_idx(scope, cnt_valid_test)
        dict_new = _get_reformatted_data_of_wikikg2(split_edge["valid"], idx_sel)
        split_edge["valid"].update(dict_new)
        valid_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="valid",
        )

        print(f"[{datetime.now()}] Reformatting test data ...")
        scope = split_edge["test"]["head"].shape[0]
        idx_sel = _get_fixed_sampled_sorted_idx(scope, cnt_valid_test)
        dict_new = _get_reformatted_data_of_wikikg2(split_edge["test"], idx_sel)
        split_edge["test"].update(dict_new)
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
        )
    else:
        dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
        )
        return dataset, [graph]


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
        )
    else:
        dataset = EnsembleNodesEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
        )
        return dataset, [graph]


@_dataset("odps_onedevice")
def _read_odps_onedevice_data(
    table,
    return_valid_test: bool = False,
    edge_dim: int = 5,
    node_dim: int = 1,
    **kwargs,
):
    slice_id = int(os.environ.get("RANK", 0))
    slice_count = int(os.environ.get("WORLD_SIZE", 1))

    y_dim = 1
    ls_tables = table.split(",")
    if return_valid_test:
        train_table, valid_table = ls_tables[:2]
        test_table = valid_table
        if len(ls_tables) == 3:
            test_table = ls_tables[2]
        train_dataset = OdpsTableIterableDataset(
            train_table,
            slice_id,
            slice_count,
            edge_dim=edge_dim,
            node_dim=node_dim,
            y_dim=y_dim,
        )
        valid_dataset = OdpsTableIterableDataset(
            valid_table,
            slice_id,
            slice_count,
            permute_nodes=False,
            edge_dim=edge_dim,
            node_dim=node_dim,
            y_dim=y_dim,
        )
        test_dataset = OdpsTableIterableDataset(
            test_table,
            slice_id,
            slice_count,
            permute_nodes=False,
            edge_dim=edge_dim,
            node_dim=node_dim,
            y_dim=y_dim,
        )
        return train_dataset, valid_dataset, test_dataset, train_dataset
    else:
        train_table = ls_tables[0]
        train_dataset = OdpsTableIterableDataset(
            train_table,
            slice_id,
            slice_count,
            edge_dim=edge_dim,
            node_dim=node_dim,
            y_dim=y_dim,
        )
        return train_dataset, train_dataset


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
def _read_odps_data(table, node_dim, edge_dim, mode="train", **kwargs):
    if mode == "train":
        slice_id = int(os.environ.get("RANK", 0))
        slice_count = int(os.environ.get("WORLD_SIZE", 1))
    else:
        slice_id = 0
        slice_count = 1
    dataset = OdpsTableIterableDataset(
        table, slice_id, slice_count, node_dim=node_dim, edge_dim=edge_dim
    )
    return dataset, dataset


def _get_global_local_id_from_onehot(x: torch.Tensor, global_id_only: bool):
    # x: one-hot tensor
    # assume each col of one-hot tensor `x` represents one species
    # each row-record belongs to one of the species

    # [N, 1] get the col-idx as the global-id, i.e., which species for the record
    global_id = torch.argmax(x, dim=-1, keepdim=True) + 1
    if global_id_only:
        output = global_id
    else:
        # [N, m] cum-sum along each col to get the cumulated count of each species
        x_cum = torch.cumsum(x, dim=0)
        # [N, m] element-wise multiplication to get the local-id of each record
        idx = x_cum * x
        local_id = idx.sum(dim=-1).view((-1, 1))  # [N, 1]
        output = torch.cat([global_id, local_id], dim=1)  # [N, 2]
    return output


def _get_global_local_id_from_enumerate(x: torch.Tensor, global_id_only: bool):
    # x: tensor of (N,) or (N,1), enumerate all possible elements
    # assume each element represents one species
    # each row-record belongs to one of the species

    # [N, 1] get the col-idx as the global-id, i.e., which species for the record
    global_id = x.view((-1, 1))
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


def _get_global_local_id_from_num_nodes(
    num_nodes: int, dividend: int, global_id_only: bool
):
    x = torch.arange(num_nodes).reshape((-1, 1))  # (N, 1)
    if global_id_only:
        output = x.clone()
    else:
        output = torch.hstack([x // dividend, x % dividend])  # (N, 2)
    return output


def _get_global_local_id_from_enumerate_with_dividend(
    x: torch.Tensor, dividend: int, global_id_only: bool = False
):
    assert len(x.shape) == 1, f"x.shape: {x.shape}"
    assert global_id_only is False
    # [N] -> [N, m]
    print("Converting to onehot ...")
    x = torch.nn.functional.one_hot(x)
    # [N, m] -> [N, 2]
    print("Getting global_local_id_from_onehot ...")
    x = _get_global_local_id_from_onehot(x, False)
    output = torch.hstack(
        [x[:, 0:1], x[:, 1:2] // dividend, x[:, 1:2] % dividend]
    )  # (N, 2) -> (N, 3)
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


def to_undirected_np(edge_index: torch.Tensor, edge_attr: torch.Tensor):
    print("Converting directed graph to un-directed graph ...")
    num_edges = edge_index.shape[1]
    edge_index = edge_index.numpy()
    bi_edge_index = np.hstack([edge_index, edge_index[[1, 0]]])
    unique_edge_index, unique_idx = np.unique(bi_edge_index, return_index=True, axis=1)

    bi_edge_type = np.hstack(
        [
            np.ones(edge_index.shape[1], dtype=int),
            np.zeros(edge_index.shape[1], dtype=int),
        ]
    )
    unique_edge_type = bi_edge_type[unique_idx].reshape((-1, 1))

    if edge_attr is not None:
        edge_attr = edge_attr.numpy()
        bi_edge_attr = np.vstack([edge_attr, edge_attr])
        unique_edge_attr = np.hstack([unique_edge_type, bi_edge_attr[unique_idx]])
    else:
        unique_edge_attr = unique_edge_type
    new_num_edges = unique_edge_index.shape[1]
    print(
        f"Finish converting directed graph to un-directed graph with num_edges {num_edges} -> {new_num_edges}"
    )
    return torch.tensor(unique_edge_index), torch.tensor(unique_edge_attr)


def to_undirected(edge_index: torch.Tensor, edge_attr: torch.Tensor):
    # compare to to_undirected_np, intend to save memory
    print("Converting directed graph to un-directed graph with pure torch ...")
    num_edges = edge_index.shape[1]
    unique_edge_index, _, unique_idx = torch_unique_with_indices(
        torch.hstack([edge_index, edge_index[[1, 0]]]), dim=1
    )

    unique_edge_type = torch.hstack(
        [
            torch.ones(num_edges, dtype=torch.int64),
            torch.zeros(num_edges, dtype=torch.int64),
        ]
    )[unique_idx].reshape((-1, 1))

    if edge_attr is not None:
        edge_attr = torch.vstack([edge_attr, edge_attr])[unique_idx]
        unique_edge_attr = torch.hstack([unique_edge_type, edge_attr])
    else:
        unique_edge_attr = unique_edge_type
    new_num_edges = unique_edge_index.shape[1]
    print(
        f"Finish converting directed graph to un-directed graph with num_edges {num_edges} -> {new_num_edges}"
    )
    return unique_edge_index, unique_edge_attr


def torch_unique_with_indices(tensor, dim=None):
    """Return the unique elements of a tensor and their indices."""
    # adapted from https://discuss.pytorch.org/t/reverse-inverse-indices-torch-unique/114521/6
    assert len(tensor.size()) <= 2, f"{len(tensor.size())} > 2"
    unique, inverse_indices = torch.unique(tensor, return_inverse=True, dim=dim)
    dim = dim or 0
    indices = torch.scatter_reduce(
        torch.zeros(tensor.size(dim), dtype=torch.long, device=tensor.device),
        dim=0,
        index=inverse_indices,
        src=torch.arange(tensor.size(dim), device=tensor.device),
        reduce="amin",
        include_self=False,
    )
    len_ = unique.size(dim)
    return unique, inverse_indices, indices[:len_]


def remove_self_cycle(edge_index: torch.Tensor, edge_attr: torch.Tensor):
    num_edges = edge_index.shape[1]
    mask = edge_index[0] != edge_index[1]
    if mask.sum().item() < num_edges:
        print(f"Removing self-cyclic node: {num_edges} -> {mask.sum().item()}")
        edge_index = edge_index[:, mask]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]
    return edge_index, edge_attr


def _get_edge_neg(source_node, target_node_neg):
    if len(source_node.shape) == 2:
        # invoke itself
        edge_neg = _get_edge_neg(target_node_neg, source_node)
        edge_neg = edge_neg[:, [1, 0]].clone()
    else:
        assert len(source_node.shape) == 1
        assert len(target_node_neg.shape) == 2
        assert source_node.shape[0] == target_node_neg.shape[0]
        num_negs = target_node_neg.shape[1]
        # [N_p] -> [N_p,1] -> [N_p, num_negs]
        source_node = source_node.reshape((-1, 1)).expand(
            source_node.shape[0], num_negs
        )
        assert source_node.shape == target_node_neg.shape
        edge_neg = torch.hstack(
            [source_node.reshape((-1, 1)), target_node_neg.reshape((-1, 1))]
        ).clone()
    return edge_neg


def _get_fixed_sampled_sorted_idx(scope, cnt_idx, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(scope, generator=g)[:cnt_idx]
    indices, _ = torch.sort(indices)
    return indices


def _get_reformatted_data_of_citation2(dict_, idx_sel):
    edge = torch.vstack(
        [
            dict_["source_node"][idx_sel],
            dict_["target_node"][idx_sel],
        ]
    ).T.clone()
    edge_neg = _get_edge_neg(
        dict_["source_node"][idx_sel],
        dict_["target_node_neg"][idx_sel],
    )
    pos_edge_attr = torch.ones((edge.shape[0], 1), dtype=torch.int64)
    neg_edge_attr = torch.ones((edge_neg.shape[0], 1), dtype=torch.int64)
    return {
        "edge": edge,
        "edge_neg": edge_neg,
        "pos_edge_attr": pos_edge_attr,
        "neg_edge_attr": neg_edge_attr,
    }


def _get_reformatted_data_of_wikikg2(dict_, idx_sel):
    edge = torch.vstack(
        [
            dict_["head"][idx_sel],
            dict_["tail"][idx_sel],
        ]
    ).T.clone()
    rel = dict_["relation"][idx_sel]
    pos_edge_attr = torch.vstack([torch.ones_like(rel), rel]).T.clone()

    edge_neg1 = _get_edge_neg(
        dict_["head"][idx_sel],
        dict_["head_neg"][idx_sel],
    )
    edge_neg2 = _get_edge_neg(
        dict_["tail_neg"][idx_sel],
        dict_["tail"][idx_sel],
    )
    edge_neg = torch.hstack([edge_neg1, edge_neg2]).reshape((-1, 2)).clone()

    neg_rel = rel.reshape((-1, 1)).expand(rel.shape[0], 1000).reshape((-1, 1))
    neg_edge_attr = torch.hstack([torch.ones_like(neg_rel), neg_rel]).clone()

    return {
        "edge": edge,
        "edge_neg": edge_neg,
        "pos_edge_attr": pos_edge_attr,
        "neg_edge_attr": neg_edge_attr,
    }
