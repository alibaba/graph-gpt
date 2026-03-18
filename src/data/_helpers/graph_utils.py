import numpy as np
import torch


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
