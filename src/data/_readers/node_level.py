import os

import torch
from datetime import datetime
from pprint import pformat
import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils.undirected import is_undirected
from omegaconf import OmegaConf
from ogb.nodeproppred import PygNodePropPredDataset
from src.conf import DataConfig, TrainingConfig
from ..dataset_map import (
    EnsembleNodesEdgesMapDataset,
    ShaDowKHopSeqMapDataset,
)
from .._helpers.node_encoding import (
    _get_global_local_id_from_onehot,
    _get_global_local_id_from_enumerate,
    _get_global_local_id_from_num_nodes,
    _get_global_local_id_from_enumerate_with_dividend,
    _mask_concat_node_label_as_feat,
)
from .._helpers.graph_utils import remove_self_cycle, to_undirected


def _read_ogbn_products(data_cfg: DataConfig, *, train_cfg: TrainingConfig, **kwargs):
    data_dir = data_cfg.data_dir
    sampling_config = OmegaConf.to_container(data_cfg.sampling, resolve=True)
    return_valid_test = data_cfg.return_valid_test
    pretrain_mode = train_cfg.pretrain_mode
    true_valid = train_cfg.ft_eval.true_valid

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


def _read_ogbn_arxiv(data_cfg: DataConfig, *, train_cfg: TrainingConfig, **kwargs):
    data_dir = data_cfg.data_dir
    sampling_config = OmegaConf.to_container(data_cfg.sampling, resolve=True)
    return_valid_test = data_cfg.return_valid_test
    pretrain_mode = train_cfg.pretrain_mode

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


def _read_ogbn_papers100M(data_cfg: DataConfig, *, train_cfg: TrainingConfig, **kwargs):
    data_dir = data_cfg.data_dir
    sampling_config = OmegaConf.to_container(data_cfg.sampling, resolve=True)
    return_valid_test = data_cfg.return_valid_test
    pretrain_mode = train_cfg.pretrain_mode
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


def _read_ogbn_proteins(data_cfg: DataConfig, *, train_cfg: TrainingConfig, **kwargs):
    data_dir = data_cfg.data_dir
    sampling_config = OmegaConf.to_container(data_cfg.sampling, resolve=True)
    return_valid_test = data_cfg.return_valid_test
    pretrain_mode = train_cfg.pretrain_mode

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


def register_node_readers(dataset_registry):
    """Register all node-level dataset readers."""
    dataset_registry("ogbn-products")(_read_ogbn_products)
    dataset_registry("ogbn-arxiv")(_read_ogbn_arxiv)
    dataset_registry("ogbn-papers100M")(_read_ogbn_papers100M)
    dataset_registry("ogbn-proteins")(_read_ogbn_proteins)
