# refer to: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
from datetime import datetime
import copy
import random
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Iterable, Callable, List
try:
    NDArray = np.typing.NDArray
except AttributeError:
    NDArray = List
from collections import defaultdict

import torch
from torch import Tensor

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.separate import separate
from torch_geometric.typing import WITH_TORCH_SPARSE, SparseTensor
from torch_geometric.utils import negative_sampling, add_self_loops
from torch_geometric.loader.cluster import ClusterData

from ..utils import control_flow
from ..utils import dataset_utils, nx_utils


_map_dataset = control_flow.Register()
init_map_dataset = _map_dataset.build  # return func/class results
get_map_dataset_class = _map_dataset.get  # return the func/class


@_map_dataset("metis")
class MetisPartitionSeqMapDataset(torch.utils.data.Dataset):
    r"""Given a graph in a :obj:`data` object, the sampler will create shallow,
    localized subgraphs with METIS partitioning.
    Each subgraph will be turned into a sequence, and then batched
    Args:
        data (torch_geometric.data.Data): The graph data object.
        sampling_config (Dict):
            num_nodes (List[int]): The approximate number of nodes in the partitioned
                subgraph.
        sample_idx (LongTensor or BoolTensor, optional): The sample index that should be
            considered for creating mini-batches.
            If set to :obj:`None`, all nodes will be
            considered.
        provide_sampler (bool, optional): If set to :obj:`True`, will provide
            an iterable sampler for DataLoader. (default: :obj:`False`)
        pretrain_mode (bool): in pretrain mode, supervised graph/node labels,
            e.g., graph.y, will be masked for valid/test data
        task_mask_func (Callable, optional): a Callable that masks input features
            for specific dataset or supervised tasks
    """

    def __init__(
        self,
        data: Data,
        sampling_config: Dict,
        *,
        sample_idx: Optional[Tensor] = None,
        provide_sampler: bool = False,
        pretrain_mode: bool = False,
        save_dir: Optional[str] = None,
        **kwargs,
    ):
        if not WITH_TORCH_SPARSE:
            raise ImportError(f"'{self.__class__.__name__}' requires 'torch-sparse'")

        assert ("id" in data) and (
            "x" in data
        ), "A big graph must have all node attrs integrated in `x`, including nodes' global id!"
        self.data = copy.copy(data)
        self.config = sampling_config["metis"]

        self.subgraph_nodes = self.config["num_nodes"]
        self.pretrain_mode = pretrain_mode
        self.save_dir = save_dir
        assert hasattr(data, "edge_index")
        self.sample_idx = sample_idx
        # 1. set-up sampler
        if provide_sampler:
            assert sample_idx is not None
            self.sampler = sample_idx.tolist()
            random.shuffle(self.sampler)
        # 2. others
        self.subgraphs = None
        self.node2subgraph = {}
        self.reset_samples_per_epoch = True
        self.reset_samples()
        self.kwargs = kwargs

    def __len__(self):
        if self.pretrain_mode:
            return len(self.subgraphs)
        else:
            return len(self.node2subgraph)

    def reset_samples(self, epoch: Optional[int] = None, seed: Optional[int] = None):
        print(f"RESET samples of {self.__class__.__name__} for epoch {epoch}!")
        epoch = 0 if epoch is None else epoch
        idx = epoch % len(self.subgraph_nodes)
        avg_num_nodes = self.subgraph_nodes[idx]
        num_parts = self.data.num_nodes // avg_num_nodes
        # 4 mins for Data(num_nodes=132534, edge_index=[2, 79122504], edge_attr=[79122504, 8], node_species=[132534, 1], y=[132534, 112], id=[132534], x=[132534, 2])
        subgraphs = ClusterData(self.data, num_parts=num_parts, save_dir=self.save_dir)
        num_subgraphs = len(subgraphs)
        self.subgraphs = [subgraph for subgraph in subgraphs if subgraph.num_nodes > 0]
        new_num_subgraphs = len(self.subgraphs)
        print(
            f"FINISH reset of {self.__class__.__name__} with {len(self.subgraphs)} subgraphs (after filtering out {num_subgraphs-new_num_subgraphs} empty subgraphs of 0 node) of average nodes {avg_num_nodes}!"
        )
        if not self.pretrain_mode:
            for subgraph in self.subgraphs:
                for i, node in enumerate(subgraph.id.tolist()):
                    tmp_graph = copy.copy(subgraph)
                    tmp_graph.seed_node = node
                    tmp_graph.root_n_id = i
                    self.node2subgraph[node] = tmp_graph
            print(
                f"FINISH reset of {self.__class__.__name__}'s dict node2subgraph with {len(self.node2subgraph)} nodes as keys and {len(self.subgraphs)} subgraphs as vals!"
            )

    def __getitem__(self, index):
        if self.pretrain_mode:
            data = self.subgraphs[index]
        else:
            data = self.node2subgraph[index]
        data.idx = index
        return index, data


@_map_dataset("node_ego")
class ShaDowKHopSeqMapDataset(torch.utils.data.Dataset):
    r"""The ShaDow :math:`k`-hop sampler from the `"Decoupling the Depth and
    Scope of Graph Neural Networks" <https://arxiv.org/abs/2201.07858>`_ paper.
    Given a graph in a :obj:`data` object, the sampler will create shallow,
    localized subgraphs.
    Each subgraph will be turned into a sequence, and then batched
    Args:
        data (torch_geometric.data.Data): The graph data object.
        sampling_config (Dict):
            depth (List[int]): The depth/number of hops of the localized subgraph.
            num_neighbors (List[int]): The number of neighbors to sample for each node in
                each hop.
            replace (bool, optional): If set to :obj:`True`, will sample neighbors
                with replacement. (default: :obj:`False`)
        adj_t (SparseTensor, optional):
        sample_idx (LongTensor or BoolTensor, optional): The sample index that should be
            considered for creating mini-batches.
            If set to :obj:`None`, all nodes will be
            considered.
        provide_sampler (bool, optional): If set to :obj:`True`, will provide
            an iterable sampler for DataLoader. (default: :obj:`False`)
        pretrain_mode (bool): in pretrain mode, supervised graph/node labels,
            e.g., graph.y, will be masked for valid/test data
        task_mask_func (Callable, optional): a Callable that masks input features
            for specific dataset or supervised tasks
    """

    def __init__(
        self,
        data: Data,
        sampling_config: Dict,
        *,
        adj_t: Optional[SparseTensor] = None,
        sample_idx: Optional[Tensor] = None,
        provide_sampler: bool = False,
        pretrain_mode: bool = False,
        task_mask_func: Optional[Callable] = None,
        **kwargs,
    ):
        if not WITH_TORCH_SPARSE:
            raise ImportError(f"'{self.__class__.__name__}' requires 'torch-sparse'")

        assert ("id" in data) and (
            "x" in data
        ), "A big graph must have all node attrs integrated in `x`, including nodes' global id!"
        self.data = copy.copy(data)
        self.config = sampling_config["node_ego"]

        self.depth_neighbors = self.config["depth_neighbors"]
        self.replace = self.config["replace"]
        self.pretrain_mode = pretrain_mode
        self.task_mask_func = task_mask_func
        # 1. set-up adj_t
        assert hasattr(data, "edge_index")
        if adj_t is None:
            row, col = data.edge_index.cpu()
            adj_t = SparseTensor(
                row=row,
                col=col,
                value=torch.arange(col.size(0)),
                sparse_sizes=(data.num_nodes, data.num_nodes),
            ).t()
        self.adj_t = adj_t
        # 2. set-up node_idx
        self._raw_node_idx = torch.arange(self.adj_t.sparse_size(0))
        if sample_idx is None:
            sample_idx = self._raw_node_idx
        self.sample_idx = sample_idx
        # 3. set-up sampler
        if provide_sampler:
            self.sampler = list(self.sample_idx.tolist())
            random.shuffle(self.sampler)
        # 4. others
        self.reset_samples()
        self.kwargs = kwargs

    def __len__(self):
        return len(self.sample_idx)

    def reset_samples(self, epoch: Optional[int] = None, seed: Optional[int] = None):
        print(f"NOT RESET samples of {self.__class__.__name__} for epoch {epoch}!")

    def __getitem__(self, index):
        assert isinstance(index, int)
        assert index >= 0
        seed_node = index
        seed_node_ids = torch.tensor(
            [seed_node], dtype=torch.int64
        )  # 1-D tensor; NOT scalar

        depth, num_neighbors = random.choice(self.depth_neighbors)
        rowptr, col, value = self.adj_t.csr()
        out = torch.ops.torch_sparse.ego_k_hop_sample_adj(
            rowptr,
            col,
            seed_node_ids,
            depth,
            num_neighbors,
            self.replace,
        )
        rowptr, col, n_id, e_id, ptr, root_n_id = out

        adj_t = SparseTensor(
            rowptr=rowptr,
            col=col,
            value=value[e_id] if value is not None else None,
            sparse_sizes=(n_id.numel(), n_id.numel()),
            is_sorted=True,
            trust_data=True,
        )

        data = Data(num_nodes=n_id.numel())
        data.root_n_id = root_n_id.item()
        data.seed_node = seed_node

        row, col, e_id = adj_t.t().coo()
        data.edge_index = torch.stack([row, col], dim=0)

        for k, v in self.data:
            if k in ["edge_index", "adj_t", "num_nodes", "batch", "ptr", "x_mask"]:
                continue
            if isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                data[k] = v[n_id]
            elif isinstance(v, Tensor) and v.size(0) == self.data.num_edges:
                data[k] = v[e_id]
            else:
                data[k] = v
        if (
            (not self.pretrain_mode)
            and hasattr(data, "x")
            and hasattr(self.data, "x_mask")
        ):
            data.x[data.root_n_id] = data.x[data.root_n_id] * self.data.x_mask
        data = self.task_mask_func(data) if self.task_mask_func is not None else data
        data.idx = index
        return index, data


@_map_dataset("edge_ego")
class ShaDowKHopSeqFromEdgesMapDataset(torch.utils.data.Dataset):
    r"""The ShaDow :math:`k`-hop sampler from the `"Decoupling the Depth and
    Scope of Graph Neural Networks" <https://arxiv.org/abs/2201.07858>`_ paper.
    Given a graph in a :obj:`data` object, the sampler will create shallow,
    localized subgraphs given two nodes.
    It can be used for link prediction tasks with algo like `SEAL`.
    Args:
        data (torch_geometric.data.Data): The graph data object.
        sampling_config (Dict):
            depth (int): The depth/number of hops of the localized subgraph.
            num_neighbors (int): The number of neighbors to sample for each node in
                each hop.
            neg_ratio (int): The ratio of negative samples to positive samples
            replace (bool, optional): If set to :obj:`True`, will sample neighbors
                with replacement. (default: :obj:`False`)
        split_edge (dict, optional):
            ```python
            from ogb.linkproppred import PygLinkPropPredDataset
            split_edge = dataset.get_edge_split()
            train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
            ```
        data_split (str): train/valid/test
    """

    def __init__(
        self,
        data: Data,
        sampling_config: Dict,
        *,
        adj_t: Optional[SparseTensor] = None,
        split_edge: Optional[Dict] = None,
        data_split: str = "train",
        pretrain_mode: bool = False,
        allow_zero_edges: bool = False,
        **kwargs,
    ):
        if not WITH_TORCH_SPARSE:
            raise ImportError(f"'{self.__class__.__name__}' requires 'torch-sparse'")

        assert ("id" in data) and (
            "x" in data
        ), "A big graph must have all node attrs integrated in `x`, including nodes' global id!"
        self.data = copy.copy(data)
        self.config = sampling_config["edge_ego"]
        # init sampling config
        self.depth_neighbors = self.config["depth_neighbors"]
        self.neg_ratio = self.config["neg_ratio"]
        self.percent = self.config.get("percent", 100)
        self.method = self.config.get("method", {"name": "global"})
        self.sample_wgt = self.config.get("sample_wgt", False)
        self.wgt = None
        # TODO: experiment on this: whether should distinguish pre-train and fine-tune
        # if pretrain_mode:
        #     self.method = {"name": "global"}
        assert 100 >= self.percent > 0
        assert isinstance(self.percent, int)
        assert self.method["name"] in {
            "local",
            "global",
        }, f"method {self.method} is NOT implemented"
        self.replace = self.config["replace"]
        # other config
        self.split_edge = split_edge
        self.data_split = data_split
        self.pretrain_mode = pretrain_mode
        assert self.data_split == "train" if self.pretrain_mode else True
        self.allow_zero_edges = allow_zero_edges
        # 1. set-up adj_t
        assert hasattr(data, "edge_index")
        if adj_t is None:
            row, col = data.edge_index.cpu()
            adj_t = SparseTensor(
                row=row,
                col=col,
                value=torch.arange(col.size(0)),
                sparse_sizes=(data.num_nodes, data.num_nodes),
            ).t()
        self.adj_t = adj_t

        self.new_edge_index, _ = add_self_loops(
            self.data.edge_index
        )  # for negative sampling only

        # obtain pos & neg edges & the labels 0/1
        if self.split_edge is None:
            print(
                "split_edge is None!!!\nSet `data_split` to be 'train', and use data.edge_index as the pos edge!!!"
            )
            self.data_split = "train"
            tmp_edge_index = self.data.edge_index.T.clone()
            mask = tmp_edge_index[:, 0] < tmp_edge_index[:, 1]
            # CAUTION: mask is for bi-directional edge!!!
            self.split_edge = {"train": {"edge": tmp_edge_index[mask]}}
            # Here .clone() must be added, otherwise will be problematic in multiprocessing
            # Check https://github.com/pyg-team/pytorch_geometric/discussions/6919
        self.dict_ = self.split_edge[self.data_split]
        self.train_count = None
        if not pretrain_mode and self.data_split == "train" and self.sample_wgt:
            # refer to: # https://github.com/snap-stanford/ogb/blob/f631af76359c9687b2fe60905557bbb241916258/examples/linkproppred/wikikg2/run.py#L190
            train_triples = self.dict_
            train_count = {}
            for head, relation, tail in tqdm(
                zip(
                    train_triples["head"].numpy(),
                    train_triples["relation"].numpy(),
                    train_triples["tail"].numpy(),
                )
            ):
                if (head, relation) not in train_count:
                    train_count[(head, relation)] = 4
                if (tail, -relation - 1) not in train_count:
                    train_count[(tail, -relation - 1)] = 4

                train_count[(head, relation)] += 1
                train_count[(tail, -relation - 1)] += 1
            self.train_count = train_count
            print(f"using sample weight with {len(self.train_count)} dict entries!!!")
        self.all_edges_with_y = None
        self.all_edge_attr = None
        self.sample_idx = None
        self.sampler = None
        self.reset_samples_per_epoch = True
        self.reset_samples()

        self.kwargs = kwargs

    def reset_samples(self, epoch: Optional[int] = 0, seed: Optional[int] = 42):
        print(
            f"[{datetime.now()}][data: {self.data_split}] RESET samples of {self.__class__.__name__} for epoch {epoch}!"
        )
        pos_edges = self.dict_["edge"]  # [N_p, 2]
        pos_edge_attr = self.dict_.get("pos_edge_attr", None)
        if "edge_neg" in self.dict_:
            neg_edges = self.dict_["edge_neg"]
            neg_edge_attr = self.dict_.get("neg_edge_attr", None)
        else:
            pos_edges, pos_edge_attr = sample_pos_edges(
                pos_edges,
                pos_edge_attr,
                percent=self.percent,
                # reset sampling
                seed=seed,
                epoch=epoch,
            )
            neg_edge_attr_candidates = self.dict_.get("neg_edge_attr_candidates", None)
            sampling_func = (
                sample_neg_edges_globally
                if self.method["name"] == "global"
                else sample_neg_edges_locally
            )
            neg_edges, neg_edge_attr = sampling_func(
                pos_edges,
                pos_edge_attr,
                # sampling params
                self_looped_edge_index=self.new_edge_index,
                num_nodes=self.data.num_nodes,
                neg_ratio=self.neg_ratio,
                neg_edge_attr_candidates=neg_edge_attr_candidates,
                method=self.method,
            )
        assert pos_edges.shape[1] == 2
        assert neg_edges.shape[1] == 2
        y_pos = torch.ones((pos_edges.shape[0], 1), dtype=torch.int64)
        y_neg = torch.zeros((neg_edges.shape[0], 1), dtype=torch.int64)
        pos_edges_with_y = torch.cat([pos_edges, y_pos], dim=1)
        neg_edges_with_y = torch.cat([neg_edges, y_neg], dim=1)
        self.all_edges_with_y = torch.cat(
            [pos_edges_with_y, neg_edges_with_y], dim=0
        )  # [N_p + N_e, 3]
        self.sample_idx = torch.arange(len(self.all_edges_with_y), dtype=torch.int64)
        self.sampler = list(self.sample_idx.tolist())
        random.shuffle(self.sampler)
        print(
            f"[{datetime.now()}] FINISH reset of {self.__class__.__name__} with {y_pos.shape[0]} pos-samples and {y_neg.shape[0]} neg-samples!\n"
        )
        if (pos_edge_attr is not None) or (neg_edge_attr is not None):
            # [N_p + N_e, edge_attr_dim]
            self.all_edge_attr = torch.cat([pos_edge_attr, neg_edge_attr], dim=0)
            assert self.all_edge_attr.shape[0] == self.all_edges_with_y.shape[0]
        if self.train_count is not None:
            # if put `defaultdict(lambda: 4)` in __init__, then error:
            # Can't pickle local object 'ShaDowKHopSeqFromEdgesMapDataset.__init__.<locals>.<lambda>'
            train_count = defaultdict(lambda: 4)
            train_count.update(self.train_count)
            ls_wgt = [
                (train_count[(head, relation)], train_count[(tail, -relation - 1)])
                for head, tail, relation in tqdm(
                    zip(
                        self.all_edges_with_y[:, 0].numpy(),
                        self.all_edges_with_y[:, 1].numpy(),
                        self.all_edge_attr[:, 1].numpy(),
                    )
                )
            ]
            arr_wgt = torch.tensor(ls_wgt).float()
            self.wgt = torch.sqrt(1 / arr_wgt.sum(dim=-1))
            assert self.wgt.shape[0] == self.all_edges_with_y.shape[0]
            print(f"top 10 ls_wgt: {ls_wgt[:10]}")

    def __len__(self):
        return len(self.all_edges_with_y)

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        edge_with_y = self.all_edges_with_y[idx]
        index = edge_with_y[:2].tolist()
        y = edge_with_y[2].view([1])
        assert len(index) == 2
        seed_node_ids = torch.tensor(index, dtype=torch.int64)  # 1-D tensor; NOT scalar

        depth, num_neighbors = random.choice(self.depth_neighbors)
        rowptr, col, _ = self.adj_t.csr()
        out = torch.ops.torch_sparse.ego_k_hop_sample_adj(
            rowptr,
            col,
            seed_node_ids,
            depth,
            num_neighbors,
            self.replace,
        )
        n_id = out[2]
        n_id_unique = torch.unique(n_id)  # The output tensor is always sorted!
        # a). func `subgraph` is too slow, change to saint_subgraph
        # edge_index, _, edge_mask = subgraph(
        #     subset=n_id_unique.to(self.device),
        #     edge_index=self.data.edge_index.to(self.device),
        #     relabel_nodes=True,
        #     num_nodes=self.data.num_nodes,
        #     return_edge_mask=True,
        # )
        # b). below use `saint_subgraph` to extract subgraph, very fast!
        adj, e_id = self.adj_t.saint_subgraph(n_id_unique)
        row, col, _ = adj.t().coo()
        edge_index = torch.vstack([row, col])
        edge_mask = e_id

        root_n_id_src = (n_id_unique == index[0]).nonzero(as_tuple=True)[0]
        root_n_id_dst = (n_id_unique == index[1]).nonzero(as_tuple=True)[0]
        root_n_id = torch.tensor([root_n_id_src, root_n_id_dst], dtype=torch.int64)

        data = Data(num_nodes=n_id_unique.numel())
        data.root_n_id = root_n_id
        data.seed_node = seed_node_ids
        if self.pretrain_mode:  # Not masking src-dst edge in pretrain mode
            non_tgt_mask = torch.tensor([True] * edge_index.shape[1])
        else:
            edge_index, non_tgt_mask = _remove_target_edge(
                edge_index, root_n_id_src, root_n_id_dst
            )
            data.y = y
        data.edge_index = edge_index

        for k, v in self.data:
            if k in ["edge_index", "adj_t", "num_nodes", "batch", "ptr", "y"]:
                continue
            if isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                data[k] = v[n_id_unique]
            elif isinstance(v, Tensor) and v.size(0) == self.data.num_edges:
                if self.allow_zero_edges and (
                    edge_mask.numel() == 0 or non_tgt_mask.numel() == 0
                ):
                    print(
                        f"[{datetime.now()}] Subgraph got no edges => [src, tgt]: {index} when assigning {k} !!!\n"
                        f"edge_mask.numel()=={edge_mask.numel()}\n"
                        f"non_tgt_mask.numel()=={non_tgt_mask.numel()}"
                    )
                    assert (
                        edge_index.numel() == 0
                    ), f"edge_index=={edge_index}\nedge_index.numel()=={edge_index.numel()}"
                    data[k] = torch.empty((0,) + v.shape[1:], dtype=v.dtype)
                else:
                    # 1st mask to obtain subgraph edges' attrs, 2nd mask to remove target edge's attrs
                    data[k] = v[edge_mask][non_tgt_mask]
            else:
                data[k] = v
        data.idx = idx
        if self.all_edge_attr is not None:
            # [N_p+N_e, edge_attr_dim] -> [edge_attr_dim]
            data.tgt_edge_attr = self.all_edge_attr[idx]
        if self.wgt is not None:
            data.wgt = self.wgt[idx]
        return idx, data


def _remove_target_edge(edge_index, src, dst, bidiretional=True):
    assert edge_index.shape[0] == 2
    forward_bool = (edge_index[0] == src) & (edge_index[1] == dst)
    if bidiretional:
        backward_bool = (edge_index[0] == dst) & (edge_index[1] == src)
        all_bool = ~(forward_bool + backward_bool)
    else:
        all_bool = ~forward_bool
    return edge_index[:, all_bool], all_bool


def sample_pos_edges(
    pos_edges,
    pos_edge_attr,
    percent,  # percentage of train samples used
    # reset sampling
    seed,
    epoch,
):
    if percent < 100:
        # logics below: for example, if `percent==10`, then epochs [0,9] will have the same seed
        # `indices` will be the same, and epochs [0,9] will use non-repeated positive edges, 10% each epoch
        # Then, from epochs [10,19], seed will be different, and will use another sequences of non-repeated positive edges.
        tot_pos_edges = pos_edges.shape[0]
        epoch_cyclic_period = int(round(100 / percent))
        seed = seed + percent * epoch // 100
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(tot_pos_edges, generator=g)
        cnt_pos_edges = int(round(tot_pos_edges * percent / 100.0))
        cyclic_epoch = epoch % epoch_cyclic_period
        pos_idx = indices[
            cyclic_epoch * cnt_pos_edges : (cyclic_epoch + 1) * cnt_pos_edges
        ]
        pos_edges = pos_edges[pos_idx]
        print(
            f"[{datetime.now()}] RESET pos_edges by sampling {cnt_pos_edges} pos edges from {tot_pos_edges} pos edges!\n"
            f"seed: {seed}, cyclic_epoch: {cyclic_epoch}, first 3 of pos_edges:\n{pos_edges[:3]}"
        )
        if pos_edge_attr is not None:
            pos_edge_attr = pos_edge_attr[pos_idx]
    return pos_edges, pos_edge_attr


def sample_neg_edges_globally(
    pos_edges,
    pos_edge_attr,
    *,
    # sampling params
    self_looped_edge_index,
    num_nodes,
    neg_ratio,
    neg_edge_attr_candidates,
    **kwargs,
):
    print(f"[{datetime.now()}] GLOBALLY sampling neg edges and edge-attrs ...")
    cnt_neg_edges = neg_ratio * pos_edges.shape[0]
    neg_edges = negative_sampling(
        edge_index=self_looped_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=cnt_neg_edges,
    ).T  # [N_e, 2]   N_e=neg_ratio*N_p
    neg_edge_attr = None
    if neg_edge_attr_candidates is not None:
        cnt_candidates = len(neg_edge_attr_candidates)
        idx_attr = torch.randint(cnt_candidates, (cnt_neg_edges,))
        # [cnt_candidates, edge_attr_dim] & [N_e] -> [N_e, edge_attr_dim]
        neg_edge_attr = neg_edge_attr_candidates[idx_attr]
    return neg_edges, neg_edge_attr


def sample_neg_edges_locally(
    pos_edges,
    pos_edge_attr,
    *,
    # sampling params
    num_nodes,
    neg_ratio,
    neg_edge_attr_candidates,
    method: Dict,
    **kwargs,
):
    # Given positive triplet, i.e., head, tail, edge (e.g., rel for KG), we can have 3 sampling strategy:
    # 1. fix `head` & `rel`, randomly sample `tail`
    # 2. fix `rel` & `tail`, randomly sample `head`
    # 3. fix `head` & `tail`, randomly sample `edge`
    print(f"[{datetime.now()}] LOCALLY sampling neg edges and edge-attrs ...")
    cnt_neg_edges = neg_ratio * pos_edges.shape[0]

    ls_neg_edges_and_attrs = []
    if method["sample_tails"]:
        ls_neg_edges_and_attrs.append(
            _local_sample_tails(
                pos_edges, pos_edge_attr, num_nodes=num_nodes, neg_ratio=neg_ratio
            )
        )
    if method["sample_heads"]:
        ls_neg_edges_and_attrs.append(
            _local_sample_heads(
                pos_edges, pos_edge_attr, num_nodes=num_nodes, neg_ratio=neg_ratio
            )
        )
    if method["sample_edges"]:
        ls_neg_edges_and_attrs.append(
            _local_sample_edges(
                pos_edges,
                pos_edge_attr,
                neg_edge_attr_candidates=neg_edge_attr_candidates,
                neg_ratio=neg_ratio,
            )
        )

    ls_neg_edges = [x[0] for x in ls_neg_edges_and_attrs]
    ls_neg_edge_attr = [x[1] for x in ls_neg_edges_and_attrs]

    neg_edge_candidates = torch.vstack(ls_neg_edges)

    # 2. select required number of neg edges and egde-attr
    g = torch.Generator()
    indices = torch.randperm(neg_edge_candidates.shape[0], generator=g)[:cnt_neg_edges]
    neg_edges = neg_edge_candidates[indices]

    neg_edge_attr = None
    if pos_edge_attr is not None:
        neg_ea_candidates = torch.vstack(ls_neg_edge_attr)
        assert neg_edge_candidates.shape[0] == neg_ea_candidates.shape[0]
        neg_edge_attr = neg_ea_candidates[indices]
    return neg_edges, neg_edge_attr


def _local_sample_tails(
    pos_edges,
    pos_edge_attr,
    *,
    num_nodes,
    neg_ratio,
):
    print(f"[{datetime.now()}] fix `head` & `edge`, randomly sample `tail` ...")
    # fix `head` & `edge`, randomly sample `tail`
    cnt_pos_edges = pos_edges.shape[0]
    cnt_neg_edges = neg_ratio * pos_edges.shape[0]
    # neg for heads, i.e., as tail of true head
    head_neg = torch.randint(num_nodes, (cnt_neg_edges, 1))
    # [Np, 1] -> [Np, nr] -> [Np*nr, 1]
    head = pos_edges[:, 0:1].expand(cnt_pos_edges, neg_ratio).reshape((-1, 1))
    # [Np*nr, 1] & [Np*nr, 1] -> [Np*nr, 2]
    neg_edges = torch.hstack([head, head_neg])

    neg_edge_attr = None
    if pos_edge_attr is not None:
        # [Np, dim] -> [Np, dim*nr] -> [Np*nr, dim]
        neg_edge_attr = torch.hstack([pos_edge_attr] * neg_ratio).reshape(
            [-1, pos_edge_attr.shape[1]]
        )
        assert (
            neg_edge_attr.shape[0] == neg_edges.shape[0]
        ), f"{neg_edge_attr.shape[0]} != {neg_edges.shape[0]}"
    return neg_edges, neg_edge_attr


def _local_sample_heads(
    pos_edges,
    pos_edge_attr,
    *,
    num_nodes,
    neg_ratio,
):
    print(f"[{datetime.now()}] fix `tail` & `edge`, randomly sample `head` ...")
    # fix `tail` & `edge`, randomly sample `head`
    cnt_pos_edges = pos_edges.shape[0]
    cnt_neg_edges = neg_ratio * pos_edges.shape[0]
    # neg for tails, i.e., as head of true tail
    tail_neg = torch.randint(num_nodes, (cnt_neg_edges, 1))
    # [Np, 1] -> [Np, nr] -> [Np*nr, 1]
    tail = pos_edges[:, 1:2].expand(cnt_pos_edges, neg_ratio).reshape((-1, 1))
    # [Np*nr, 1] & [Np*nr, 1] -> [Np*nr, 2]
    neg_edges = torch.hstack([tail_neg, tail])

    neg_edge_attr = None
    if pos_edge_attr is not None:
        # [Np, dim] -> [Np, dim*nr] -> [Np*nr, dim]
        neg_edge_attr = torch.hstack([pos_edge_attr] * neg_ratio).reshape(
            [-1, pos_edge_attr.shape[1]]
        )
        assert (
            neg_edge_attr.shape[0] == neg_edges.shape[0]
        ), f"{neg_edge_attr.shape[0]} != {neg_edges.shape[0]}"
    return neg_edges, neg_edge_attr


def _local_sample_edges(
    pos_edges,
    pos_edge_attr,
    *,
    neg_edge_attr_candidates,
    neg_ratio,
):
    print(f"[{datetime.now()}] fix `head` & `tail`, randomly sample `edge` ...")
    assert len(pos_edges.shape) == 2
    assert pos_edges.shape[1] == 2

    neg_edges = torch.empty((0, 2), dtype=pos_edges.dtype)
    neg_edge_attr = None
    if pos_edge_attr is not None:
        neg_edges = torch.hstack([pos_edges] * neg_ratio).reshape([-1, 2])
        pos_edge_attr = torch.hstack([pos_edge_attr] * neg_ratio).reshape(
            [-1, pos_edge_attr.shape[1]]
        )
        neg_edge_attr = _get_edge_attr_neg(pos_edge_attr, neg_edge_attr_candidates)
        assert (
            neg_edges.shape[0] == neg_edge_attr.shape[0]
        ), f"{neg_edges.shape[0]} != {neg_edge_attr.shape[0]}"
    return neg_edges, neg_edge_attr


def _get_edge_attr_neg(pos_edge_attr, neg_edge_attr_candidates):
    cnt_pos_edges = pos_edge_attr.shape[0]
    cnt_candidates = len(neg_edge_attr_candidates)
    assert (
        cnt_candidates > 2
    ), f"NOT implemented for cnt_candidates ({cnt_candidates}) <= 2"

    idx_attr1 = torch.randint(cnt_candidates, (cnt_pos_edges,))
    neg_edge_attr1 = neg_edge_attr_candidates[idx_attr1]

    idx_attr2 = torch.randint(cnt_candidates, (cnt_pos_edges,))
    neg_edge_attr2 = neg_edge_attr_candidates[idx_attr2]

    mask1 = _get_row_equal_mask(pos_edge_attr, neg_edge_attr1)
    neg_edge_attr = (~mask1).view((-1, 1)).to(
        torch.int64
    ) * neg_edge_attr1 + mask1.view((-1, 1)).to(torch.int64) * neg_edge_attr2
    mask = _get_row_equal_mask(pos_edge_attr, neg_edge_attr)
    print(
        f"[WARNING] {mask.sum().item()} out of {neg_edge_attr.shape[0]} neg-edge-attr is the same as pos-edge-attr"
    )
    return neg_edge_attr


def _get_row_equal_mask(a, b):
    # return mask tensor of shape (a.shape[0],), True if the row is equal
    assert (
        len(a.shape) == len(b.shape) == 2
    ), f"a -> {len(a.shape)}, b -> {len(b.shape)}"
    return ~torch.abs(a - b).sum(dim=-1).to(bool)


@_map_dataset("edge_random")
class RandomEdgesMapDataset(torch.utils.data.Dataset):
    r"""Random Edges Partition.
    Args:
        data (torch_geometric.data.Data): The graph data object.
        sampling_config (Dict):
            num_edges (int): The number of edges per subgraph approximately.
        split_edge (dict, optional):
            ```python
            from ogb.linkproppred import PygLinkPropPredDataset
            split_edge = dataset.get_edge_split()
            train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
            ```
        data_split (str): train/valid/test
    """

    def __init__(
        self,
        data: Data,
        sampling_config: Dict,
        *,
        adj_t: Optional[SparseTensor] = None,
        split_edge: Optional[Dict] = None,
        data_split: str = "train",
        pretrain_mode: bool = False,
        **kwargs,
    ):
        if not WITH_TORCH_SPARSE:
            raise ImportError(f"'{self.__class__.__name__}' requires 'torch-sparse'")

        assert ("id" in data) and (
            "x" in data
        ), "A big graph must have all node attrs integrated in `x`, including nodes' global id!"
        self.data = copy.copy(data)
        self.config = sampling_config["edge_random"]
        # init sampling config
        self.num_edges = self.config["num_edges"]
        self.neg_ratio = self.config["neg_ratio"]
        self.percent = self.config.get("percent", 100)
        assert 100 >= self.percent > 0
        assert isinstance(self.percent, int)
        # other config
        self.split_edge = split_edge
        self.data_split = data_split
        self.pretrain_mode = pretrain_mode
        assert self.data_split == "train" if self.pretrain_mode else True
        # 1. set-up adj_t
        assert hasattr(data, "edge_index")
        if adj_t is None:
            row, col = data.edge_index.cpu()
            adj_t = SparseTensor(
                row=row,
                col=col,
                value=torch.arange(col.size(0)),
                sparse_sizes=(data.num_nodes, data.num_nodes),
            ).t()
        self.adj_t = adj_t

        self.new_edge_index, _ = add_self_loops(
            self.data.edge_index
        )  # for negative sampling only

        # obtain pos & neg edges & the labels 0/1
        if self.split_edge is None:
            print(
                "split_edge is None!!!\nSet `data_split` to be 'train', and use data.edge_index as the pos edge!!!"
            )
            self.data_split = "train"
            tmp_edge_index = self.data.edge_index.T.clone()
            mask = tmp_edge_index[:, 0] < tmp_edge_index[:, 1]
            self.split_edge = {"train": {"edge": tmp_edge_index[mask]}}
            # Here .clone() must be added, otherwise will be problematic in multiprocessing
            # Check https://github.com/pyg-team/pytorch_geometric/discussions/6919
        self.dict_ = self.split_edge[self.data_split]
        self.all_edges_with_y = None
        self.ls_edges = None
        self.ls_subgraphs = None
        self.reset_samples_per_epoch = True
        self.reset_samples()

        self.kwargs = kwargs

    def reset_samples(self, epoch: Optional[int] = None, seed: Optional[int] = None):
        print(f"RESET samples of {self.__class__.__name__} for epoch {epoch}!")
        pos_edges = self.dict_["edge"]  # [N_p, 2]
        if "edge_neg" in self.dict_:
            neg_edges = self.dict_["edge_neg"]
        else:
            if self.percent < 100:
                cnt_pos_edges = int(round(pos_edges.shape[0] * self.percent / 100.0))
                pos_idx = torch.randint(pos_edges.shape[0], (cnt_pos_edges,))
                pos_edges = pos_edges[pos_idx]
                print(
                    f"RESET pos_edges by sampling {cnt_pos_edges} pos edges from {self.dict_['edge'].shape[0]} pos edges!"
                )
            else:
                cnt_pos_edges = pos_edges.shape[0]
            cnt_neg_edges = self.neg_ratio * cnt_pos_edges
            neg_edges = negative_sampling(
                edge_index=self.new_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=cnt_neg_edges,
            ).T.to(
                torch.int64
            )  # [N_e, 2]   N_e=neg_ratio*N_p
        assert pos_edges.shape[1] == 2
        assert neg_edges.shape[1] == 2
        y_pos = torch.ones((pos_edges.shape[0], 1), dtype=torch.int64)
        y_neg = torch.zeros((neg_edges.shape[0], 1), dtype=torch.int64)
        pos_edges_with_y = torch.cat([pos_edges, y_pos], dim=1)
        neg_edges_with_y = torch.cat([neg_edges, y_neg], dim=1)
        self.all_edges_with_y = torch.cat(
            [pos_edges_with_y, neg_edges_with_y], dim=0
        )  # [N_p + N_e, 3]
        tot_edges = self.all_edges_with_y.shape[0]
        idx_all = torch.randperm(tot_edges)
        self.ls_edges = []
        self.ls_subgraphs = []
        for i in tqdm(range(0, tot_edges, self.num_edges)):
            idx = idx_all[i : i + self.num_edges]
            edges_with_y = self.all_edges_with_y[idx]  # [num_edges, 3]
            nodes = torch.unique(edges_with_y[:, :2])
            if self.pretrain_mode:
                self.ls_subgraphs.append(nodes)
            else:
                self.ls_edges.extend([(t, nodes.clone()) for t in edges_with_y])
        print(
            f"FINISH reset of {self.__class__.__name__} with {y_pos.shape[0]} pos-samples and {y_neg.shape[0]} neg-samples and {len(self.ls_subgraphs)} subgraphs and {len(self.ls_edges)} edges!"
        )

    def __len__(self):
        if self.pretrain_mode:
            return len(self.ls_subgraphs)
        else:
            return len(self.ls_edges)

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        if self.pretrain_mode:
            n_id_unique = self.ls_subgraphs[idx]
        else:
            edge_with_y, n_id_unique = self.ls_edges[idx]
        data = Data(num_nodes=n_id_unique.numel())
        # below use `saint_subgraph` to extract subgraph, very fast!
        adj, e_id = self.adj_t.saint_subgraph(n_id_unique)
        row, col, _ = adj.t().coo()
        edge_index = torch.vstack([row, col])
        edge_mask = e_id

        if self.pretrain_mode:  # Not masking src-dst edge in pretrain mode
            non_tgt_mask = torch.tensor([True] * edge_index.shape[1])
        else:
            index = edge_with_y[:2].tolist()
            y = edge_with_y[2].view([1])
            assert len(index) == 2
            # set-up seed_node & root_n_id
            seed_node_ids = torch.tensor(
                index, dtype=torch.int64
            )  # 1-D tensor; NOT scalar
            root_n_id_src = (n_id_unique == index[0]).nonzero(as_tuple=True)[0]
            root_n_id_dst = (n_id_unique == index[1]).nonzero(as_tuple=True)[0]
            root_n_id = torch.tensor([root_n_id_src, root_n_id_dst], dtype=torch.int64)

            data.root_n_id = root_n_id
            data.seed_node = seed_node_ids

            edge_index, non_tgt_mask = _remove_target_edge(
                edge_index, root_n_id_src, root_n_id_dst
            )
            data.y = y

        data.edge_index = edge_index

        for k, v in self.data:
            if k in ["edge_index", "adj_t", "num_nodes", "batch", "ptr", "y"]:
                continue
            if isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                data[k] = v[n_id_unique]
            elif isinstance(v, Tensor) and v.size(0) == self.data.num_edges:
                # 1st mask to obtain subgraph edges' attrs, 2nd mask to remove target edge's attrs
                data[k] = v[edge_mask][non_tgt_mask]
            else:
                data[k] = v
        data.idx = idx
        return idx, data


@_map_dataset("node_random")
class RandomNodesMapDataset(torch.utils.data.Dataset):
    r"""Randomly select some nodes to generate a subgraph
    Args:
        data (torch_geometric.data.Data): The graph data object.
        sampled_nodes (int): The number of nodes to be sampled to construct a subgraph.
    """

    def __init__(
        self,
        data: Data,
        sampling_config: Dict,
        *,
        adj_t: Optional[SparseTensor] = None,
        sample_idx: Optional[Tensor] = None,
        provide_sampler: bool = False,
        **kwargs,
    ):
        if not WITH_TORCH_SPARSE:
            raise ImportError(f"'{self.__class__.__name__}' requires 'torch-sparse'")

        assert ("id" in data) and (
            "x" in data
        ), "A big graph must have all node attrs integrated in `x`, including nodes' global id!"
        self.data = copy.copy(data)
        self.config = sampling_config["node_random"]

        self.sampled_nodes = self.config["sampled_nodes"]
        # 1. set-up adj_t
        assert hasattr(data, "edge_index")
        if adj_t is None:
            row, col = data.edge_index.cpu()
            adj_t = SparseTensor(
                row=row,
                col=col,
                value=torch.arange(col.size(0)),
                sparse_sizes=(data.num_nodes, data.num_nodes),
            ).t()
        self.adj_t = adj_t
        # 2. set-up node_idx
        self._raw_node_idx = torch.arange(self.adj_t.sparse_size(0))
        if sample_idx is None:
            sample_idx = self._raw_node_idx
        self.sample_idx = sample_idx
        # 3. set-tup sampler
        if provide_sampler:
            self.sampler = list(self.sample_idx.tolist())
            random.shuffle(self.sampler)
        # 4. others
        self.samples = None
        self.reset_samples_per_epoch = True
        self.reset_samples()
        self.kwargs = kwargs

    def reset_samples(self, epoch: Optional[int] = None, seed: Optional[int] = None):
        print(f"RESET samples of {self.__class__.__name__} for epoch {epoch}!")
        tgt_nodes = torch.arange(self.data.num_nodes, dtype=torch.int64).view(
            1, -1
        )  # [1, N]
        random_nodes = torch.randint(
            self.data.num_nodes,
            (self.sampled_nodes, self.data.num_nodes),
            dtype=torch.int64,
        )  # [sampled_nodes, N]
        self.samples = torch.cat(
            [tgt_nodes, random_nodes], dim=0
        )  # [sampled_nodes+1, N]

    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        n_id = self.samples[:, idx]  # [sampled_nodes+1]
        n_id_unique = torch.unique(n_id)  # The output tensor is always sorted!
        # b). below use `saint_subgraph` to extract subgraph, very fast!
        adj, e_id = self.adj_t.saint_subgraph(n_id_unique)
        row, col, _ = adj.t().coo()
        edge_index = torch.vstack([row, col])
        edge_mask = e_id

        seed_node_id = n_id[0].item()
        root_n_id = (n_id_unique == seed_node_id).nonzero(as_tuple=True)[0].item()

        data = Data(num_nodes=n_id_unique.numel())
        data.root_n_id = root_n_id
        data.seed_node = seed_node_id
        data.edge_index = edge_index

        for k, v in self.data:
            if k in ["edge_index", "adj_t", "num_nodes", "batch", "ptr"]:
                continue
            if isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                data[k] = v[n_id_unique]
            elif isinstance(v, Tensor) and v.size(0) == self.data.num_edges:
                data[k] = v[edge_mask]
            else:
                data[k] = v
        data.idx = idx
        return idx, data


class EnsembleNodesEdgesMapDataset(torch.utils.data.Dataset):
    r"""Randomly select some nodes to generate a subgraph
    Args:
        data (torch_geometric.data.Data): The graph data object.
        sampling_config (Dict): The config for different sampling strategies.
    """

    def __init__(
        self,
        data: Data,
        sampling_config: Dict,
        *,
        adj_t: Optional[SparseTensor] = None,
        sample_idx: Optional[Tensor] = None,
        provide_sampler: bool = False,
        **kwargs,
    ):
        if not WITH_TORCH_SPARSE:
            raise ImportError(f"'{self.__class__.__name__}' requires 'torch-sparse'")

        assert ("id" in data) and (
            "x" in data
        ), "A big graph must have all node attrs integrated in `x`, including nodes' global id!"
        self.data = copy.copy(data)
        self.config = sampling_config

        # 1. set-up adj_t
        assert hasattr(data, "edge_index")
        if adj_t is None:
            row, col = data.edge_index.cpu()
            adj_t = SparseTensor(
                row=row,
                col=col,
                value=torch.arange(col.size(0)),
                sparse_sizes=(data.num_nodes, data.num_nodes),
            ).t()
        self.adj_t = adj_t
        # 2. set-up node_idx
        self._raw_node_idx = torch.arange(self.adj_t.sparse_size(0))
        if sample_idx is None:
            sample_idx = self._raw_node_idx
        self.sample_idx = sample_idx
        # 3. set-up sampler
        if provide_sampler:
            self.sampler = list(self.sample_idx.tolist())
            random.shuffle(self.sampler)

        self.ls_dataset = []
        for key in self.config.keys():
            if self.config[key]["valid"]:
                print(f"Init dataset-map: {key}")
                dataset = init_map_dataset(
                    key,
                    data,
                    sampling_config,
                    adj_t=self.adj_t,
                    sample_idx=sample_idx,
                    provide_sampler=provide_sampler,
                    **kwargs,
                )
                self.ls_dataset.append(dataset)
        self.samples = None
        self.reset_samples()
        self.kwargs = kwargs

    def reset_samples(self, epoch: Optional[int] = None, seed: Optional[int] = None):
        for dataset in self.ls_dataset:
            dataset.reset_samples(epoch)

    def __len__(self):
        lens = [len(ds) for ds in self.ls_dataset]
        assert min(lens) == max(lens)
        return lens[0]

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        dataset = random.choice(self.ls_dataset)
        return dataset[idx]


class GraphsMapDataset(torch.utils.data.Dataset):
    r"""MapStyleDataset for many small graphs
    Args:
        data: The torch-geometric `InMemoryDataset`.
        sampling_config: dummy
        sample_idx (Tensor):
        permute_nodes (bool): whether to permute node index. This serve as a kind of data argumentation for graph-gpt,
            especially for structural learning
        provide_sampler (bool):
        with_prob (bool): whether to generate samples based on the number of Eulerian paths
        ensemble_paths (bool): tried, seems useless in improving results; on the other hand, it proves that
            the model do understand the different eulerian paths of the same graph and yield consistent predictions
    """

    def __init__(
        self,
        data: InMemoryDataset,
        sampling_config: Optional[Dict],
        *,
        sample_idx: Optional[Tensor] = None,
        permute_nodes: bool = True,
        provide_sampler: bool = False,
        with_prob: bool = False,
        ensemble_paths: bool = False,
        shift_distribution: bool = False,
        **kwargs,
    ):
        self.cls = data._data.__class__
        self.data = data._data
        self.slice_dict = data.slices
        self.num_graphs = len(data)
        self.permute_nodes = permute_nodes
        self.g = None
        self.idx_of_ds = 0
        # cannot pickle 'torch._C.Generator' object, so self.g has to None
        if self.permute_nodes:
            print(
                "[Warning] permute_nodes enabled! edge_attr remains the same; edge_index and node-attrs will be affected!\n"
                * 5
            )
        # 1. set-up sampler
        self.ensemble_paths = ensemble_paths
        if self.ensemble_paths:
            # try Tensor.scatter_reduce_ to groupby the results
            assert (
                with_prob is False
            ), "ensemble_paths and with_prob cannot be True at the same time!"
            self.gn_idx = _get_graph_node_idx(data, sample_idx)  # [2, num_graph_nodes]
            self.sample_idx = torch.arange(self.gn_idx.shape[1])
            print(
                f"Under ensemble_paths, samples to infer increases: {len(sample_idx)} -> {len(self.sample_idx)}"
            )
        else:
            self.sample_idx = sample_idx
        self.with_prob = with_prob
        self.shift_distribution = shift_distribution
        if provide_sampler:
            assert self.sample_idx is not None
            if self.with_prob:
                self.graph_wgts = dataset_utils.obtain_graph_wgts(
                    dataset=data, idx=sample_idx
                )
                self.sampler = self._generate_samples_with_prob()
            elif self.shift_distribution:
                self.vec_dist, self.min_num_nodes = _get_target_distribution(data)
                self.dict_num_nodes2indices = _get_candidate_graphs_mapping(
                    data, sample_idx
                )
                self.sampler = shift_to_target_distribution(
                    train_num=len(self.sample_idx),
                    min_num_nodes=self.min_num_nodes,
                    distribution=self.vec_dist,
                    dict_num_nodes2indices=self.dict_num_nodes2indices,
                    seed=0,
                )
            else:
                self.sampler = self.sample_idx.tolist()
            random.shuffle(self.sampler)
            self.num_graphs = len(self.sampler)
        # 2. other config
        self.reset_samples()
        # 3. temporary code, for experiemtal only. # TODO: refactor this
        if (
            hasattr(self.data, "y")
            and self.data.y is not None
            and len(self.data.y.shape) == 2
            and self.data.y.shape[1] > 1
        ):
            self.idx_tuple = tuple(list(range(self.data.y.shape[1])))
        else:
            self.idx_tuple = None
        # TODO: causing problems in ogbg-molpcba dataset, so disable it!
        self.idx_tuple = None
        print(f"idx_tuple: {self.idx_tuple}")

        self.kwargs = kwargs

    def _generate_samples_with_prob(self):
        picked_graph = np.random.choice(
            self.sample_idx, size=len(self.sample_idx), replace=True, p=self.graph_wgts
        )
        return picked_graph.tolist()

    def reset_samples(self, epoch: Optional[int] = None, seed: int = 0):
        if self.with_prob:
            self.sampler = self._generate_samples_with_prob()
            print(
                f"[{datetime.now()}] RESET samples of {self.__class__.__name__} of {self.num_graphs} graphs with prob for epoch {epoch}!\nObtaining {len(np.unique(self.sampler))} unique graphs"
            )
        elif self.shift_distribution:
            self.sampler = shift_to_target_distribution(
                train_num=len(self.sample_idx),
                min_num_nodes=self.min_num_nodes,
                distribution=self.vec_dist,
                dict_num_nodes2indices=self.dict_num_nodes2indices,
                seed=epoch + seed if epoch is not None else seed,
            )
            self.sampler = sorted(self.sampler)
            num = len(self.sampler) // 2
            print(
                f"[{datetime.now()}] RESET samples of {self.__class__.__name__} of {self.num_graphs} graphs with shifted distribution for epoch {epoch}!"
                f"\nObtaining {len(self.sampler)} total graphs"
                f"\n{len(np.unique(self.sampler))} unique graphs"
                f"\nself.sampler[{num}:{num+10}] -> {self.sampler[num:num+10]}"
            )
        else:
            self.sampler = sorted(self.sampler)
            print(
                f"[{datetime.now()}] NOT RESET samples of {self.__class__.__name__} of {self.num_graphs} graphs for epoch {epoch}!"
            )

    def get_random_sample_idx(self):
        # [WARNING] Dataloader with multiple workers, i.e. multi-processing will produce idx out of range;
        # To avoid this, use all torch tensor as var inheriting from parent process
        # refer to: https://stackoverflow.com/questions/72794398/np-random-choice-conflict-with-multiprocessing-multiprocessing-inside-for-loop
        # and to: https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
        if self.with_prob:
            idx = np.random.choice(self.sample_idx, p=self.graph_wgts).item()
        else:
            idx = np.random.choice(self.sample_idx).item()
        return idx

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        if self.ensemble_paths:
            idx, node_idx = self.gn_idx[:, idx].tolist()
        # refer to: torch_geometric/data/in_memory_dataset.py/InMemoryDataset:: `def get`
        graph = separate(
            cls=self.cls,
            batch=self.data,
            idx=idx,
            slice_dict=self.slice_dict,
            decrement=False,
        )
        if self.permute_nodes:
            graph, _ = nx_utils.permute_nodes(graph, self.g)
        graph.idx = idx
        if self.idx_tuple:
            idx_correction = random.choice(self.idx_tuple)
            graph.y = graph.y[:, idx_correction : idx_correction + 1].reshape((-1))
        else:
            idx_correction = 0
        graph.idx_of_ds = self.idx_of_ds + idx_correction
        if self.ensemble_paths:
            # TODO: permute nodes may affect this, investigate it!
            graph.root_n_id = node_idx
        return idx, graph


def _get_graph_node_idx(dataset: InMemoryDataset, graph_idx: torch.Tensor):
    assert hasattr(dataset, "x")
    x_idx = dataset.slices["x"]
    # cnt of nodes of all graphs
    cnt_nodes = x_idx[1:] - x_idx[:-1]
    cnt_nodes_specific = cnt_nodes[graph_idx]

    def _merge_gn_idx(g_idx, num_nodes):
        return torch.tensor([[g_idx] * num_nodes, list(range(num_nodes))])

    ls_tensor = [
        _merge_gn_idx(g_idx, num_nodes)
        for g_idx, num_nodes in tqdm(zip(graph_idx, cnt_nodes_specific))
    ]
    gn_idx = torch.cat(ls_tensor, dim=-1)
    return gn_idx


def _get_target_distribution(dataset: InMemoryDataset):
    # obtain distribution p(num_nodes) of `valid` & `test-dev` graphs for PCQM4M-v2
    x_idx = dataset.slices["x"]
    vec_num_nodes = x_idx[1:] - x_idx[:-1]

    dict_idx = dataset.get_idx_split()
    target_idx = torch.cat([dict_idx["valid"], dict_idx["test-dev"]])
    target_num_nodes = vec_num_nodes[target_idx].numpy()

    max_num_nodes = target_num_nodes.max().item()
    min_num_nodes = target_num_nodes.min().item()

    values, counts = np.unique(target_num_nodes, return_counts=True)

    eps = 0.1
    vec_dist = np.zeros(max_num_nodes - min_num_nodes + 1) + eps
    for val, cnt in zip(values, counts):
        val_idx = val - min_num_nodes
        vec_dist[val_idx] = vec_dist[val_idx] + cnt
    vec_dist = vec_dist / vec_dist.sum()
    return vec_dist, min_num_nodes


def _get_candidate_graphs_mapping(dataset: InMemoryDataset, train_idx: Tensor):
    # obtain `dict_num_nodes2indices`
    # key is `num_nodes`, val is `list of graph's index`
    x_idx = dataset.slices["x"]
    vec_num_nodes = x_idx[1:] - x_idx[:-1]
    train_num_nodes = vec_num_nodes[train_idx]
    dict_num_nodes2indices = {}
    for num_nodes, idx in zip(train_num_nodes.numpy(), train_idx.numpy()):
        if num_nodes not in dict_num_nodes2indices:
            dict_num_nodes2indices[num_nodes] = []
        dict_num_nodes2indices[num_nodes].append(idx)
    dict_num_nodes2indices = {k: np.array(v) for k, v in dict_num_nodes2indices.items()}
    return dict_num_nodes2indices


def shift_to_target_distribution(
    train_num: int,
    min_num_nodes: int,
    distribution: NDArray,
    dict_num_nodes2indices: Dict,
    seed: int,
):
    # set `seed` to ensure the same results across different workers
    print(f"using seed {seed} to shift distribution")
    rng = np.random.default_rng(seed)
    cnts = (train_num * distribution).round().astype(int)

    all_idx = []
    for raw_i, cnt in enumerate(cnts):
        raw_num_nodes = raw_i + min_num_nodes
        num_nodes = raw_num_nodes
        num_shift = 0
        # If given `num_nodes` has no graphs in train data, then shift the `num_nodes` to its neighbors
        while num_nodes not in dict_num_nodes2indices:
            num_shift = num_shift + 1
            if raw_num_nodes + num_shift in dict_num_nodes2indices:
                num_nodes = raw_num_nodes + num_shift
            elif raw_num_nodes - num_shift in dict_num_nodes2indices:
                num_nodes = raw_num_nodes - num_shift
            else:
                assert (
                    num_shift < 30
                ), f"num_shift: {num_shift} too large, break the loop!"
        ls_candidate_index = dict_num_nodes2indices[num_nodes]  # np.array
        if len(ls_candidate_index) >= cnt:
            ls_chosen_index = rng.choice(
                ls_candidate_index, size=cnt, replace=False
            ).tolist()
        else:
            ls_chosen_index = (
                rng.choice(
                    ls_candidate_index, size=cnt - len(ls_candidate_index), replace=True
                ).tolist()
                + ls_candidate_index.tolist()
            )
        assert (
            len(ls_chosen_index) == cnt
        ), f"len(ls_chosen_index): {len(ls_chosen_index)} != cnt: {cnt}"
        all_idx.extend(ls_chosen_index)
    return all_idx


class EnsembleGraphsMapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: List[GraphsMapDataset],
    ):
        self.datasets = datasets
        ls_double_idx = []
        print(f"Combining {len(datasets)} datasets into one Ensemble")
        for i, ds in enumerate(self.datasets):
            num = len(ds.sample_idx)
            idx_of_ds = torch.tensor([i] * num, dtype=torch.int64)
            double_idx = torch.vstack([idx_of_ds, ds.sample_idx]).T  # [2, num]
            ls_double_idx.append(double_idx)
            ds.idx_of_ds = i
        self.all_idx = torch.vstack(ls_double_idx)
        self.num_graphs = len(self.all_idx)
        self.sample_idx = torch.arange(self.num_graphs, dtype=torch.int64)
        self.sampler = self.sample_idx.tolist()
        random.shuffle(self.sampler)
        self.reset_samples()

    def reset_samples(self, epoch: Optional[int] = None, seed: Optional[int] = None):
        for ds in self.datasets:
            ds.reset_samples(epoch, seed)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        idx_of_ds, idx = self.all_idx[idx].tolist()
        return self.datasets[idx_of_ds][idx]
