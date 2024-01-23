import copy
import random
import numpy as np
import torch
import base64

from typing import Optional
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.typing import WITH_TORCH_SPARSE, SparseTensor

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


class OdpsTableIterableDataset(torch.utils.data.IterableDataset):
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

        table_start, table_end = _get_slice_range(
            self.row_count, worker_id, num_workers, self.start_pos
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
                    edge_index = torch.from_numpy(
                        np.frombuffer(
                            base64.b64decode(data[0]), dtype=np.int64
                        ).reshape(2, -1)
                    )
                    edge_attr = torch.from_numpy(
                        np.frombuffer(
                            base64.b64decode(data[1]), dtype=np.int64
                        ).reshape(edge_index.shape[1], -1)
                    )
                    x = torch.from_numpy(
                        np.frombuffer(
                            base64.b64decode(data[2]), dtype=np.int64
                        ).reshape(-1, 9)
                    )
                    y = torch.from_numpy(
                        np.frombuffer(base64.b64decode(data[3]), dtype=np.float32)
                    )
                    new_data = {
                        "edge_index": edge_index,
                        "edge_attr": edge_attr,
                        "x": x,
                        "y": y,
                    }

                except common_io.exception.OutOfRangeException:
                    reader.close()
                    break
                yield Data.from_dict(new_data)

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
