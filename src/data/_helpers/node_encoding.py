import torch
from torch_geometric.data import Data


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
