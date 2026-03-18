import os

import torch
from datetime import datetime
from pprint import pformat

from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from ogb.graphproppred import PygGraphPropPredDataset
from src.utils import control_flow, dataset_utils
from ..conf import DataConfig, TrainingConfig
from .dataset_map import (
    GraphsMapDataset,
    EnsembleGraphsMapDataset,
)

from .dataset_iterable import (
    GraphsIterableDataset,
    OdpsTableIterableDataset,
    OdpsTableIterableDatasetOneID,
)

from ._graph_factory import DatasetSpec, register_specs


_dataset = control_flow.Register()
read_dataset = _dataset.build  # return func results
get_dataset_reader = _dataset.get  # return the func

_molecule = control_flow.Register()


def read_merge_molecule_datasets(data_cfg: DataConfig):
    ls_edge_attr = []
    ls_x = []
    for ds in _molecule._register_map.keys():
        print(f"load molecule dataset {ds}!\n")
        _, raw_dataset = read_dataset(ds, data_cfg)
        ls_edge_attr.append(raw_dataset._data.edge_attr)
        ls_x.append(raw_dataset._data.x)
    data = Data(edge_attr=torch.vstack(ls_edge_attr), x=torch.vstack(ls_x))
    print(f"Merged molecule dataset:\n{data}")
    return [data]


@_dataset("structure")
def _read_structure(data_cfg: DataConfig, *, with_prob: bool = False, **kwargs):
    data_dir = data_cfg.data_dir
    return_valid_test = data_cfg.return_valid_test
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
def _read_random_graph_structure(data_cfg: DataConfig, **kwargs):
    print(f"\n[{datetime.now()}] Generating dataset randomly ...")
    dataset = GraphsIterableDataset(
        num_nodes_low=10,
        num_nodes_high=101,
        edges_per_node=4,
    )
    return dataset, dataset


@_dataset("molecule")
def _read_molecules(data_cfg: DataConfig, **kwargs):
    ensemble_datasets = data_cfg.ensemble_datasets
    ls_ds = []
    for ds_name in ensemble_datasets:
        dataset, raw_dataset = read_dataset(ds_name, data_cfg, pt_all=True)
        ls_ds.append(dataset)
    train_dataset = EnsembleGraphsMapDataset(ls_ds)
    return train_dataset, raw_dataset


# ---------------------------------------------------------------------------
# DatasetSpec-driven graph-level readers
# ---------------------------------------------------------------------------


def _triangles_label_transform(dataset):
    dataset._data.y = dataset._data.y.to(dtype=torch.int64) - 1


def _cepdb_post_load(dataset):
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


def _zinc_post_load(dataset):
    dataset._data.y = None  # tmp for pre-training


_GRAPH_SPECS = [
    DatasetSpec(
        name="spice-circuit",
        dataset_cls=dataset_utils.SpiceCircuitDataset,
        dataset_kwargs={"data_ver": "v2"},
    ),
    DatasetSpec(
        name="oneid",
        dataset_cls=dataset_utils.OneIDSmallDataset,
        valid_slice=slice(None, 100000),
        test_slice=slice(-100000, None),
        test_from_valid=True,
    ),
    DatasetSpec(
        name="triangles",
        dataset_cls=TUDataset,
        dataset_kwargs={"name": "TRIANGLES"},
        split_method="hardcoded",
        hardcoded_splits={
            "train": slice(None, 30000),
            "valid": slice(35000, 40000),
            "test": slice(40000, 45000),
        },
        ft_permute_nodes=True,
        pretrain_permute_nodes=True,
        label_transform=_triangles_label_transform,
    ),
    DatasetSpec(
        name="reddit_threads",
        dataset_cls=TUDataset,
        dataset_kwargs={"name": "reddit_threads"},
        split_method="random",
        random_train_ratio=0.8,
    ),
    DatasetSpec(
        name="ogbg-molhiv",
        dataset_cls=PygGraphPropPredDataset,
        dataset_kwargs={"name": "ogbg-molhiv"},
        also_molecule=True,
        pretrain_permute_nodes=True,
    ),
    DatasetSpec(
        name="ogbg-molpcba",
        dataset_cls=PygGraphPropPredDataset,
        dataset_kwargs={"name": "ogbg-molpcba"},
        also_molecule=True,
        pretrain_permute_nodes=True,
        pretrain_sample_idx="train_split",
    ),
    DatasetSpec(
        name="CEPDB",
        dataset_cls=dataset_utils.PygCEPDBDataset,
        also_molecule=True,
        pretrain_only=True,
        post_load_hook=_cepdb_post_load,
    ),
    DatasetSpec(
        name="ZINC",
        dataset_cls=dataset_utils.PygZINCDataset,
        dataset_kwargs={"subset": 11},
        also_molecule=True,
        pretrain_only=True,
        post_load_hook=_zinc_post_load,
    ),
    DatasetSpec(
        name="custom_mol",
        dataset_cls=dataset_utils.PygCustomMolDataset,
        pretrain_only=True,
        pretrain_sample_idx=lambda ds: torch.cat(
            [torch.arange(1), torch.arange(len(ds))]
        ),
    ),
]

register_specs(_GRAPH_SPECS, _dataset, _molecule)


# ---------------------------------------------------------------------------
# PCQM4Mv2 reader (extracted to _readers/pcqm4mv2.py)
# ---------------------------------------------------------------------------
from ._readers.pcqm4mv2 import register_pcqm4mv2
register_pcqm4mv2(_dataset, _molecule)


# ---------------------------------------------------------------------------
# Node-level readers (extracted to _readers/node_level.py)
# ---------------------------------------------------------------------------
from ._readers.node_level import register_node_readers
register_node_readers(_dataset)


# ---------------------------------------------------------------------------
# Edge-level readers (extracted to _readers/edge_level.py)
# ---------------------------------------------------------------------------
from ._readers.edge_level import register_edge_readers
register_edge_readers(_dataset)


@_dataset("odps_onedevice")
def _read_odps_onedevice_data(data_cfg: DataConfig, **kwargs):
    table = data_cfg.odps.tables
    return_valid_test = data_cfg.return_valid_test
    edge_dim = data_cfg.odps.edge_dim
    node_dim = data_cfg.odps.node_dim

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
def _read_odps_oneid_data(data_cfg: DataConfig, **kwargs):
    table = data_cfg.odps.tables
    return_valid_test = data_cfg.return_valid_test
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
def _read_odps_data(data_cfg: DataConfig, **kwargs):
    table = data_cfg.odps.tables
    edge_dim = data_cfg.odps.edge_dim
    node_dim = data_cfg.odps.node_dim
    mode = data_cfg.odps.mode
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


# ---------------------------------------------------------------------------
# Re-exported helpers (moved to _helpers/ sub-package)
# ---------------------------------------------------------------------------
from ._helpers.node_encoding import (  # noqa: F401
    _get_global_local_id_from_onehot,
    _get_global_local_id_from_enumerate,
    _get_global_local_id_from_num_nodes,
    _get_global_local_id_from_enumerate_with_dividend,
    _mask_concat_node_label_as_feat,
)
from ._helpers.graph_utils import (  # noqa: F401
    to_undirected_np,
    to_undirected,
    torch_unique_with_indices,
    remove_self_cycle,
)
from ._helpers.edge_formatting import (  # noqa: F401
    _get_edge_neg,
    _get_fixed_sampled_sorted_idx,
    _get_reformatted_data_of_citation2,
    _get_reformatted_data_of_wikikg2,
)
