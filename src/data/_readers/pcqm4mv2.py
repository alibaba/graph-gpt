import os
import time

import torch
from typing import List
import numpy as np

from tqdm import tqdm
from ogb.lsc import PygPCQM4Mv2Dataset
from src.utils import dataset_utils, mol_utils
from src.conf import DataConfig, TrainingConfig
from ..dataset_map import (
    GraphsMapDataset,
    EnsembleGraphsMapDataset,
)


def _read_pcqm4mv2(
    data_cfg: DataConfig,
    *,
    train_cfg: TrainingConfig,
    with_prob: bool = False,
    pt_all: bool = False,  # whether to use all data in pre-train
    **kwargs,
):
    data_dir = data_cfg.data_dir
    return_valid_test = data_cfg.return_valid_test
    true_valid = train_cfg.ft_eval.true_valid
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
    dataset, pt_all = PygPCQM4Mv2Dataset(root=data_dir), True  # official version
    # dataset._data -> Data(edge_index=[2, 109093626], edge_attr=[109093626, 3], x=[52970652, 9], y=[3746620])
    # dataset = dataset_utils.PygPCQM4Mv2ExtraDataset(root=data_dir)
    # dataset._data -> Data(edge_index=[2, 109093626], edge_attr=[109093626, 3], x=[52970652, 12], y=[3746620])
    # dataset, pt_all = (
    #     dataset_utils.PygChembl29Dataset(root=data_dir),
    #     True,
    # )  # temporary, for testing ONLY
    pretrain_idx = torch.arange(len(dataset))
    # dataset._data -> Data(edge_index=[2, 137821426], edge_attr=[137821426, 3], x=[63727450, 9], y=[2084723])
    print(f"\ndataset._data -> {dataset._data}")

    if isinstance(dataset, dataset_utils.PygPCQM4Mv2ExtraDataset):
        dataset._data.x = dataset._data.x[:, [0, 4, 5, 9, 10, 11]]
        dataset._data.edge_attr = dataset._data.edge_attr[:, [1]]
        print(f"\ndataset._data -> {dataset._data}")

    # BELOW 3 lines are for testing generation capability
    if train_cfg.do_generation:
        dataset._data.x = dataset._data.x[:, 0:1]
        dataset._data.edge_attr = dataset._data.edge_attr[:, 0:1]
        print(f"\nMol Gen Pretraining!!!\ndataset._data -> {dataset._data}")

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
    if return_valid_test:
        split_idx = dataset.get_idx_split()
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
        if train_cfg.ft_eval.save_hidden_states:
            print("\nLoading Custom Mol dataset ...")
            dataset_custom = dataset_utils.PygCustomMolDataset(root=data_dir)
            print(f"\ndataset_custom._data -> {dataset_custom._data}")
            # dataset_custom._data -> Data(edge_index=[2, 502532], edge_attr=[502532, 3], x=[231956, 9], y=[8269])
            test_dataset = GraphsMapDataset(
                dataset_custom,
                None,
                sample_idx=torch.arange(len(dataset_custom)),
                provide_sampler=True,
                with_prob=with_prob,
            )
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
        # ls_idx = obtain_special_molecules(dataset)
        # try:
        #     fn = "dedup_idx" if pt_all else "dedup_idx_train_valid"
        #     while not os.path.exists(os.path.join(dataset.root, fn)):
        #         if int(dist.get_rank()) == 0:
        #             _generate_dedup_idx(dataset, fn, pt_all)
        #         else:
        #             seconds = 3
        #             print(f"sleep for {seconds} seconds ...")
        #             time.sleep(seconds)
        #     pretrain_idx = _load_idx_from_file(dataset.root, fn)
        #     print(
        #         f"Using dedup_idx with {len(pretrain_idx)} molecules instead of original {len(dataset)} molecules!"
        #     )
        # except Exception as inst:
        #     print(inst)
        #     pretrain_idx = remove_special_molecules(torch.arange(len(dataset)), ls_idx)
        if not pt_all:
            split_idx = dataset.get_idx_split()
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


# ---------------------------------------------------------------------------
# PCQM4Mv2-specific helper functions
# ---------------------------------------------------------------------------

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


def register_pcqm4mv2(dataset_registry, molecule_registry):
    """Register PCQM4Mv2 reader into both registries."""
    dataset_registry("PCQM4Mv2")(_read_pcqm4mv2)
    molecule_registry("PCQM4Mv2")(_read_pcqm4mv2)
