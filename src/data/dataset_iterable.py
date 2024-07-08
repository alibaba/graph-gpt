import copy
import random
import numpy as np
import torch
import base64

from typing import Optional, Dict
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.typing import WITH_TORCH_SPARSE, SparseTensor
from torch_geometric.utils import erdos_renyi_graph

from ..utils import nx_utils

# import common_io


class ShaDowKHopSeqIterDataset(torch.utils.data.IterableDataset):
    r"""The ShaDow :math:`k`-hop sampler from the `"Decoupling the Depth and
    Scope of Graph Neural Networks" <https://arxiv.org/abs/2201.07858>`_ paper.
    Given a graph in a :obj:`data` object, the sampler will create shallow,
    localized subgraphs.
    Each subgraph will be turned into a sequence, and then batched
    Args:
        data (torch_geometric.data.Data): The graph data object.
        depth (int): The depth/number of hops of the localized subgraph.
        num_neighbors (int): The number of neighbors to sample for each node in
            each hop.
        node_idx (LongTensor or BoolTensor, optional): The nodes that should be
            considered for creating mini-batches.
            If set to :obj:`None`, all nodes will be
            considered.
        replace (bool, optional): If set to :obj:`True`, will sample neighbors
            with replacement. (default: :obj:`False`)
    """

    def __init__(
        self,
        data: Data,
        depth: int,
        num_neighbors: int,
        node_idx: Optional[Tensor] = None,
        replace: bool = False,
        **kwargs,
    ):
        if not WITH_TORCH_SPARSE:
            raise ImportError(f"'{self.__class__.__name__}' requires 'torch-sparse'")

        assert ("id" in data) and (
            "x" in data
        ), "A big graph must have all node attrs integrated in `x`, including nodes' global id!"
        self.data = copy.copy(data)
        self.depth = depth
        self.num_neighbors = num_neighbors
        self.replace = replace

        if data.edge_index is not None:
            self.is_sparse_tensor = False
            row, col = data.edge_index.cpu()
            self.adj_t = SparseTensor(
                row=row,
                col=col,
                value=torch.arange(col.size(0)),
                sparse_sizes=(data.num_nodes, data.num_nodes),
            ).t()
        else:
            self.is_sparse_tensor = True
            self.adj_t = data.adj_t.cpu()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)
        self.node_idx = node_idx.tolist()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        assert worker_id < num_workers
        random.seed(42 + worker_id)
        samples_per_worker = len(self.node_idx) // num_workers
        local_idx = [
            num_workers * i + worker_id for i in range(samples_per_worker)
        ]  # TODO: optimize it because tail is dropped
        for _ in range(
            samples_per_worker
        ):  # TODO: change to `while True` to enable streaming
            seed_node = random.choice(local_idx)
            seed_node_ids = torch.tensor([seed_node])  # 1-D tensor; NOT scalar

            rowptr, col, value = self.adj_t.csr()
            out = torch.ops.torch_sparse.ego_k_hop_sample_adj(
                rowptr, col, seed_node_ids, self.depth, self.num_neighbors, self.replace
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

            if self.is_sparse_tensor:
                data.adj_t = adj_t
            else:
                row, col, e_id = adj_t.t().coo()
                data.edge_index = torch.stack([row, col], dim=0)

            for k, v in self.data:
                if k in ["edge_index", "adj_t", "num_nodes", "batch", "ptr"]:
                    continue
                if isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                    data[k] = v[n_id]
                elif isinstance(v, Tensor) and v.size(0) == self.data.num_edges:
                    data[k] = v[e_id]
                else:
                    data[k] = v

            yield data


class GraphsIterableDataset(torch.utils.data.IterableDataset):
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
        num_nodes_low: int = 5,
        num_nodes_high: int = 101,
        edges_per_node: int = 4,
        **kwargs,
    ):
        self.vec_num_nodes = np.array(range(num_nodes_low, num_nodes_high))
        vec_num_nodes_cnt = np.log(self.vec_num_nodes)
        self.node_prob = vec_num_nodes_cnt / vec_num_nodes_cnt.sum()
        self.edge_prob = {
            num_node: edges_per_node / (num_node - 1) for num_node in self.vec_num_nodes
        }
        print(
            f"num_nodes_low: {num_nodes_low}\nnum_nodes_high: {num_nodes_high}\nedges_per_node:{edges_per_node}"
        )
        print(
            f"vec_num_nodes: {self.vec_num_nodes}\nnode_prob: {self.node_prob}\nedge_prob:{self.edge_prob}"
        )

        self.reset_samples()
        self.kwargs = kwargs

    def reset_samples(self, epoch: Optional[int] = None):
        print(
            f"NOT RESET samples of {self.__class__.__name__} of infinite graphs for epoch {epoch}!"
        )

    def __iter__(self):
        def random_graph_iterator():
            while True:
                num_nodes = np.random.choice(self.vec_num_nodes, p=self.node_prob)
                p = self.edge_prob[num_nodes]
                edge_index = erdos_renyi_graph(num_nodes, edge_prob=p, directed=False)
                graph = Data(edge_index=edge_index, num_nodes=num_nodes, idx=0)
                # idx for downstream compatibility
                yield 0, graph

        return random_graph_iterator()

    def __len__(self):
        return int(1e12)


class OdpsTableIterableDatasetOneID(torch.utils.data.IterableDataset):
    def __init__(
        self,
        table_path,
        slice_id=0,
        slice_count=1,
        **kwargs,
    ):
        self.table_path = table_path
        reader = common_io.table.TableReader(
            table_path, slice_id=slice_id, slice_count=slice_count, num_threads=0
        )
        self.row_count = reader.get_row_count()
        self.start_pos = reader.start_pos
        self.end_pos = reader.end_pos
        reader.close()
        super(OdpsTableIterableDatasetOneID, self).__init__()
        print(
            "table total_row_count:{}, start_pos:{}, end_pos:{}".format(
                self.row_count, self.start_pos, self.end_pos
            )
        )
        self.data_ver = "v9"
        dict_edge_attr_dim = {
            "v2": 2,
            "v3": 5,
            "v4": 6,
            "v5": 5,
            "v6": 5,
            "v7": 2,
            "v8": 2,
            "v9": 6,
        }
        self.edge_attr_dim = dict_edge_attr_dim[self.data_ver]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # print(f"worker_info:{worker_info}")
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        print("worker_id:{}, num_workers:{}".format(worker_id, num_workers))

        table_start, table_end = _get_slice_range(
            self.row_count, worker_id, num_workers, self.start_pos
        )
        table_path = "{}?start={}&end={}".format(
            self.table_path, table_start, table_end
        )
        print("table_path:%s" % table_path)

        def table_data_iterator():
            reader = common_io.table.TableReader(
                table_path, num_threads=2, capacity=10000
            )
            while True:
                try:
                    dp_val = reader.read(num_records=1, allow_smaller_final_batch=True)[
                        0
                    ]
                    x = torch.tensor(
                        np.frombuffer(
                            base64.b64decode(dp_val[2]), dtype=np.int64
                        ).reshape([-1, 1])
                    )
                    edge_index = torch.tensor(
                        np.frombuffer(
                            base64.b64decode(dp_val[0]), dtype=np.int64
                        ).reshape([2, -1])
                    )
                    edge_attr = torch.tensor(
                        np.frombuffer(base64.b64decode(dp_val[1]), dtype=np.int64)
                        .reshape([self.edge_attr_dim, -1])
                        .T
                    )
                    a2d = torch.tensor(
                        np.frombuffer(base64.b64decode(dp_val[4]), dtype=np.int64)
                        .reshape([2, -1])
                        .T
                    )
                    graph = Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        a2d=a2d,
                        num_nodes=len(x),
                        key_type=torch.LongTensor([int(dp_val[7])]),
                    )

                except common_io.exception.OutOfRangeException:
                    reader.close()
                    break
                yield 0, graph

        return table_data_iterator()

    def __len__(self):
        return self.row_count


class OdpsTableIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        table_path,
        slice_id=0,
        slice_count=1,
        skipped_samples=0,
        permute_nodes=True,
        epoch=0,
    ):
        self.epoch = epoch
        self.slice_id = slice_id
        # skipped_samples per-gpu, in case of resuming training
        self.skipped_samples = skipped_samples
        self.permute_nodes = permute_nodes
        self.table_path = table_path
        reader = common_io.table.TableReader(
            table_path, slice_id=slice_id, slice_count=slice_count, num_threads=0
        )
        self.row_count = reader.get_row_count()
        self.start_pos = reader.start_pos
        self.end_pos = reader.end_pos
        reader.close()
        super(OdpsTableIterableDataset, self).__init__()
        print(
            "table total_row_count:{}, start_pos:{}, end_pos:{}".format(
                self.row_count, self.start_pos, self.end_pos
            )
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # print(f"worker_info:{worker_info}")
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        # print("worker_id:{}, num_workers:{}".format(worker_id, num_workers))
        mapping = list(range(num_workers))
        random.seed(self.epoch * 1000 + self.slice_id)
        random.shuffle(mapping)
        mapped_worker_id = mapping[worker_id]
        # above 4 lines to ensure partial randomness when reading the same odps table in different epochs

        table_start, table_end = _get_slice_range(
            self.row_count, mapped_worker_id, num_workers, self.start_pos
        )
        table_start_tmp = table_start
        skipped_samples_per_worker = self.skipped_samples // num_workers
        table_start += skipped_samples_per_worker
        print(
            f"table start for worker_id {worker_id}: {table_start_tmp} -> {table_start}\nwith skipped samples-per-worker: {skipped_samples_per_worker} out of total skipped samples {self.skipped_samples}"
        )
        table_path = "{}?start={}&end={}".format(
            self.table_path, table_start, table_end
        )
        # print("table_path:%s" % table_path)

        def table_data_iterator():
            reader = common_io.table.TableReader(
                table_path, num_threads=4, capacity=40000
            )
            while True:
                try:
                    data = reader.read(num_records=1, allow_smaller_final_batch=True)[0]
                    # cols: smiles, edge_index, edge_feat, node_feat, num_nodes
                    edge_index = torch.from_numpy(
                        np.frombuffer(
                            base64.urlsafe_b64decode(data[1]), dtype=np.int64
                        ).reshape(2, -1)
                    )
                    edge_attr = torch.from_numpy(
                        np.frombuffer(
                            base64.urlsafe_b64decode(data[2]), dtype=np.int64
                        ).reshape(-1, 3)
                    )
                    x = torch.from_numpy(
                        np.frombuffer(
                            base64.urlsafe_b64decode(data[3]), dtype=np.int64
                        ).reshape(-1, 9)
                    )
                    # y = torch.from_numpy(
                    #     np.frombuffer(base64.b64decode(data[3]), dtype=np.float32)
                    # )
                    new_data = {
                        "edge_index": edge_index,
                        "edge_attr": edge_attr,
                        "x": x,
                        # "y": y,
                    }
                    graph = Data.from_dict(new_data)
                    if self.permute_nodes:
                        graph, _ = nx_utils.permute_nodes(graph, None)

                except common_io.exception.OutOfRangeException:
                    reader.close()
                    break
                yield 0, graph

        return table_data_iterator()

    def __len__(self):
        return self.row_count


class OdpsTableIterableTokenizedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        table_path,
        slice_id=0,
        slice_count=1,
        supervised_task="graph",  # graph|node|edge
        **kwargs,
    ):
        self.table_path = table_path
        reader = common_io.table.TableReader(
            self.table_path, slice_id=slice_id, slice_count=slice_count, num_threads=0
        )
        self.row_count = reader.get_row_count()
        self.start_pos = reader.start_pos
        self.end_pos = reader.end_pos
        reader.close()
        self.supervised_task = supervised_task
        super(OdpsTableIterableTokenizedDataset, self).__init__()
        print(
            "table total_row_count:{}, start_pos:{}, end_pos:{}".format(
                self.row_count, self.start_pos, self.end_pos
            )
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # print(f"worker_info:{worker_info}")
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        # print("worker_id:{}, num_workers:{}".format(worker_id, num_workers))

        table_start, table_end = _get_slice_range(
            self.row_count, worker_id, num_workers, self.start_pos
        )
        table_path = "{}?start={}&end={}".format(
            self.table_path, table_start, table_end
        )
        # print("table_path:%s" % table_path)

        def table_data_iterator():
            reader = common_io.table.TableReader(
                table_path, num_threads=4, capacity=16384
            )
            while True:
                try:
                    data = reader.read(num_records=1, allow_smaller_final_batch=True)[0]
                    idx = str(data[0])  # data[0] is bytes
                    input_ids = np.frombuffer(
                        base64.urlsafe_b64decode(data[1]), dtype=np.int64
                    ).tolist()
                    labels = np.frombuffer(
                        base64.urlsafe_b64decode(data[2]), dtype=np.int64
                    ).tolist()
                    task_labels = (
                        data[3]
                        if isinstance(data[3], int)
                        else np.frombuffer(
                            base64.urlsafe_b64decode(data[3]), dtype=np.int64
                        ).tolist()
                    )
                    attention_mask = np.ones_like(input_ids, dtype=np.int64).tolist()
                    position_ids = list(range(len(input_ids)))
                    in_dict = {
                        # "idx": idx,
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        f"{self.supervised_task}_labels": task_labels,
                    }

                except common_io.exception.OutOfRangeException:
                    reader.close()
                    break
                yield in_dict

        return table_data_iterator()

    def __len__(self):
        return self.row_count


# below copied from https://code.alibaba-inc.com/deep_algo/oneid/blob/IDMappingE2E/data/oneid_mtp_dataset.py
# with modifications:
# 1. `self._get_slice_range` -> _get_slice_range
# 2. method `pad`
class OdpsIterableDatasetMTP(torch.utils.data.IterableDataset):
    def __init__(self, table_path, slice_id=0, slice_count=1, max_len_input=256):
        self.table_path = table_path
        reader = common_io.table.TableReader(
            table_path, slice_id=slice_id, slice_count=slice_count, num_threads=0
        )
        self.row_count = reader.get_row_count()
        self.start_pos = reader.start_pos
        self.end_pos = reader.end_pos
        self.max_len_input = max_len_input
        reader.close()
        super(OdpsIterableDatasetMTP, self).__init__()
        print(
            "table total_row_count:{}, start_pos:{}, end_pos:{}".format(
                self.row_count, self.start_pos, self.end_pos
            )
        )

    def __len__(self):
        return self.row_count

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        print("worker_id:{}, num_workers:{}".format(worker_id, num_workers))

        table_start, table_end = _get_slice_range(
            self.row_count, worker_id, num_workers, self.start_pos
        )
        table_path = "{}?start={}&end={}".format(
            self.table_path, table_start, table_end
        )
        print("table_path:%s" % table_path)

        def table_data_iterator():
            reader = common_io.table.TableReader(
                table_path, num_threads=1, capacity=1024
            )
            while True:
                try:
                    data = reader.read(num_records=1, allow_smaller_final_batch=False)
                except common_io.exception.OutOfRangeException:
                    reader.close()
                    break

                start_node = str(data[0][1], "utf-8")
                re_dict_map_str = str(data[0][13], "utf-8")

                pe = list(map(int, str(data[0][3], "utf-8").split(",")))
                se = [x + 1 for x in map(int, str(data[0][4], "utf-8").split(","))]
                re = [x for x in map(int, str(data[0][5], "utf-8").split(","))]
                edge_token_pv = [
                    min(x, 100000000)
                    for x in map(int, str(data[0][6], "utf-8").split(","))
                ] + [100000000]
                edge_token_date = [
                    min(x, 90) for x in map(int, str(data[0][7], "utf-8").split(","))
                ] + [91]

                st_pe = list(map(int, str(data[0][8], "utf-8").split(",")))
                st_se = [x + 1 for x in map(int, str(data[0][9], "utf-8").split(","))]
                pr_re = [x for x in map(int, str(data[0][12], "utf-8").split(","))]

                pr_pe = list(map(int, str(data[0][10], "utf-8").split(",")))
                pr_se = [x + 1 for x in map(int, str(data[0][11], "utf-8").split(","))]

                pe = pe + st_pe
                se = se + st_se
                re = re + pr_re

                in_dict = {
                    "se_ids": se,
                    "pe_ids": pe,
                    "re_ids": re,
                    "edge_token_pv": edge_token_pv,
                    "edge_token_date": edge_token_date,
                    "st_pe_ids": torch.LongTensor(st_pe),
                    "st_se_ids": torch.LongTensor(st_se),
                    "pr_pe_ids": torch.LongTensor(pr_pe),
                    "pr_se_ids": torch.LongTensor(pr_se),
                    "pr_re_ids": torch.LongTensor(pr_re),
                    "len": len(se),
                    "start_node": start_node,
                    "re_dict_map_str": re_dict_map_str,
                }

                pad_in_dict = self.pad(self.max_len_input, in_dict)
                yield pad_in_dict

        return table_data_iterator()

    def pad(self, max_len, feature):  # compatible with graph-gpt's stack tokenizer
        in_dict = feature
        node_id = in_dict["pe_ids"]
        edge_type = [ele + 200 for ele in in_dict["re_ids"]]
        edge_pv = [min(ele, 10240) + 400 for ele in in_dict["edge_token_pv"]]

        vocab = 10240 + 400
        attention_mask = [1] * len(node_id)
        input_ids = [list(ele) for ele in list(zip(node_id, edge_type, edge_pv))]
        dict_ = {
            "node_labels": in_dict["pr_pe_ids"].item(),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": list(range(len(input_ids))),
            "labels": list(node_id[1:]) + [-100],
        }
        return dict_

    def _pad(self, max_len, feature):  # original by xinshuai
        if max_len > len(feature["se_ids"]):
            padding_len = max_len - len(feature["se_ids"])
            padded_input_ids = [0] * padding_len
            feature["se_ids"] = torch.LongTensor(
                feature["se_ids"][:-1] + padded_input_ids + [feature["se_ids"][-1]]
            )
            feature["re_ids"] = torch.LongTensor(
                feature["re_ids"][:-1] + padded_input_ids + [feature["re_ids"][-1]]
            )
            feature["pe_ids"] = torch.LongTensor(
                feature["pe_ids"][:-1] + padded_input_ids + [feature["pe_ids"][-1]]
            )
            feature["edge_token_pv"] = torch.LongTensor(
                feature["edge_token_pv"][:-1]
                + padded_input_ids
                + [feature["edge_token_pv"][-1]]
            )
            feature["edge_token_date"] = torch.LongTensor(
                feature["edge_token_date"][:-1]
                + padded_input_ids
                + [feature["edge_token_date"][-1]]
            )

            # 创建一个由1和0组成的列表，其长度与padded_input_ids相同，1对应实际token，0对应填充
            feature["attention_mask"] = torch.LongTensor(
                [1] * (max_len - padding_len - 1) + [0] * padding_len + [1, 1]
            )  # sepecial_token+pred+token
        else:
            feature["se_ids"] = torch.LongTensor(
                feature["se_ids"][: max_len - 1] + [feature["se_ids"][-1]]
            )
            feature["re_ids"] = torch.LongTensor(
                feature["re_ids"][: max_len - 1] + [feature["re_ids"][-1]]
            )
            feature["pe_ids"] = torch.LongTensor(
                feature["pe_ids"][: max_len - 1] + [feature["pe_ids"][-1]]
            )
            feature["edge_token_pv"] = torch.LongTensor(
                feature["edge_token_pv"][: max_len - 1] + [feature["edge_token_pv"][-1]]
            )
            feature["edge_token_date"] = torch.LongTensor(
                feature["edge_token_date"][: max_len - 1]
                + [feature["edge_token_date"][-1]]
            )
            feature["attention_mask"] = torch.LongTensor([1] * (max_len + 1))
        return feature


def _get_slice_range(row_count, worker_id, num_workers, baseline=0):
    # div-mod split, each slice data count max diff 1
    # print(f"row:{row_count}, id:{worker_id}, workers:{num_workers}")
    size = int(row_count / num_workers)
    split_point = row_count % num_workers
    if worker_id < split_point:
        start = worker_id * (size + 1) + baseline
        end = start + (size + 1)
    else:
        start = split_point * (size + 1) + (worker_id - split_point) * size + baseline
        end = start + size
    return start, end


def get_odps_writer(table_name, slice_id):
    return common_io.table.TableWriter(table_name, slice_id=slice_id)
