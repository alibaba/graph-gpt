import torch
from datetime import datetime
from pprint import pformat
import numpy as np

from torch_geometric.data import Data
from omegaconf import OmegaConf
from ogb.linkproppred import PygLinkPropPredDataset
from src.conf import DataConfig, TrainingConfig
from ..dataset_map import (
    EnsembleNodesEdgesMapDataset,
    ShaDowKHopSeqFromEdgesMapDataset,
)
from .._helpers.node_encoding import (
    _get_global_local_id_from_onehot,
    _get_global_local_id_from_num_nodes,
    _get_global_local_id_from_enumerate_with_dividend,
)
from .._helpers.graph_utils import remove_self_cycle, to_undirected
from .._helpers.edge_formatting import (
    _get_fixed_sampled_sorted_idx,
    _get_reformatted_data_of_citation2,
    _get_reformatted_data_of_wikikg2,
)


def _read_ogbl_ppa(data_cfg: DataConfig, *, train_cfg: TrainingConfig, **kwargs):
    data_dir = data_cfg.data_dir
    sampling_config = OmegaConf.to_container(data_cfg.sampling, resolve=True)
    return_valid_test = data_cfg.return_valid_test
    true_valid = train_cfg.ft_eval.true_valid
    pretrain_mode = train_cfg.pretrain_mode

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
        if true_valid > 0:
            valid_dataset.reset_samples_per_epoch = False
            sampler = np.sort(valid_dataset.sampler)
            rng = np.random.default_rng(seed=true_valid)
            sampled_elements = rng.choice(sampler, size=true_valid, replace=False)
            valid_dataset.sampler = sampled_elements.tolist()
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


def _read_ogbl_citation2(data_cfg: DataConfig, *, train_cfg: TrainingConfig, **kwargs):
    data_dir = data_cfg.data_dir
    sampling_config = OmegaConf.to_container(data_cfg.sampling, resolve=True)
    return_valid_test = data_cfg.return_valid_test
    pretrain_mode = train_cfg.pretrain_mode
    true_valid = train_cfg.ft_eval.true_valid

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


def _read_ogbl_wikikg2(data_cfg: DataConfig, *, train_cfg: TrainingConfig, **kwargs):
    data_dir = data_cfg.data_dir
    sampling_config = OmegaConf.to_container(data_cfg.sampling, resolve=True)
    return_valid_test = data_cfg.return_valid_test
    pretrain_mode = train_cfg.pretrain_mode

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


def _read_ogbl_ddi(data_cfg: DataConfig, *, train_cfg: TrainingConfig, **kwargs):
    data_dir = data_cfg.data_dir
    sampling_config = OmegaConf.to_container(data_cfg.sampling, resolve=True)
    return_valid_test = data_cfg.return_valid_test
    pretrain_mode = train_cfg.pretrain_mode

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


def register_edge_readers(dataset_registry):
    """Register all edge/link-level dataset readers."""
    dataset_registry("ogbl-ppa")(_read_ogbl_ppa)
    dataset_registry("ogbl-citation2")(_read_ogbl_citation2)
    dataset_registry("ogbl-wikikg2")(_read_ogbl_wikikg2)
    dataset_registry("ogbl-ddi")(_read_ogbl_ddi)
