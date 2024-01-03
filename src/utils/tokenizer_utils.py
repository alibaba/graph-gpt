import networkx as nx
import random
import itertools
import torch
from typing import List, Union, Tuple, Dict, Iterable
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from . import control_flow

TASK_TYPES = {"pretrain", "node", "edge", "graph", "graph-dh"}
ATTR_ASSIGNMENT_TYPES = {"first", "last", "random", "all"}

_inputs_deco = control_flow.Register()
prepare_inputs_for_task = _inputs_deco.build  # return func results
get_inputs_preparation_func = _inputs_deco.get  # return the func


def get_edge_index(edge_index: Tensor, src: int, tgt: int) -> Tensor:
    vec_bool = (edge_index[0, :] == src) & (edge_index[1, :] == tgt)
    idx = vec_bool.nonzero(as_tuple=True)[0]  # tensor
    return idx


def get_edge_type(edge_index: Tensor, src: int, tgt: int):
    foward_idx = get_edge_index(edge_index, src, tgt)
    backward_idx = get_edge_index(edge_index, tgt, src)
    if foward_idx.shape[0] == 0:
        if backward_idx.shape[0] == 0:
            edge_type = "<edge_jump>"
        else:
            edge_type = "<edge_in>"
    else:
        if backward_idx.shape[0] == 0:
            edge_type = "<edge_out>"
        else:
            edge_type = "<edge_bi>"
    return edge_type


def connect_graph_central(G):
    if not nx.is_connected(G):
        jump_edges = []
        components = [
            tuple(com) for com in list(nx.connected_components(G))
        ]  # list of tuples of nodes
        # random.shuffle(components)
        main_component = components[0]
        for com in components[1:]:
            src = random.choice(main_component)
            tgt = random.choice(com)
            jump_edges.append((src, tgt))
            jump_edges.append((tgt, src))
        G.add_edges_from(jump_edges)
    return G


def connect_graph_sequential(G):
    if not nx.is_connected(G):
        jump_edges = []
        components = [
            tuple(com) for com in list(nx.connected_components(G))
        ]  # list of tuples of nodes
        random.shuffle(components)
        for src_comp, tgt_comp in zip(components[:-1], components[1:]):
            src = random.choice(src_comp)
            tgt = random.choice(tgt_comp)
            jump_edges.append((src, tgt))
            jump_edges.append((tgt, src))
        G.add_edges_from(jump_edges)
    return G


def connect_graph(G):
    return connect_graph_sequential(G)
    # return connect_graph_central(G)


def shorten_path(path):
    """
    If the given path is euler path, then it will go back to the start node, meaning that some edges are duplicated after
    all edges have been visited. So we need to remove those unnecessary edges.
    If the given path is semi-euler path, then usually there is no unnecessarily repeated edges.
    :param path:
    :return:
    """
    triangle_path = [(src, tgt) if src < tgt else (tgt, src) for src, tgt in path]
    unique_edges = set(triangle_path)
    idx = 0
    for i in range(1, len(path) + 1):
        short_path = triangle_path[:i]
        if set(short_path) == unique_edges:
            idx = i
            break
    path = path[:idx]
    return path


def graph2path(graph: Data, prioritize: bool = False) -> List[Tuple[int]]:
    G = to_networkx(graph, to_undirected="upper").to_undirected()
    # 1. Eulerize the graph if it is not
    G = connect_graph(G)
    # if not (nx.is_eulerian(G) or nx.is_semieulerian(G)):
    # Eulerize semi-eulerian graph, too; otherwise ONLY two paths is available -> NOT enough regularization.
    if not nx.is_eulerian(G):
        G = nx.eulerize(G)
    # 2. loop through nodes, and get one euler path if exists
    g = list(G.nodes())
    random.shuffle(g)
    if prioritize and hasattr(graph, "root_n_id"):
        root_n_id = graph.root_n_id
        assert isinstance(root_n_id, torch.Tensor) or isinstance(root_n_id, int)
        root_n_id = (
            root_n_id.tolist() if isinstance(root_n_id, torch.Tensor) else [root_n_id]
        )
        random.shuffle(root_n_id)
        [g.remove(x) for x in root_n_id]
        g = root_n_id + g  # prioritize path starting from the target nodes!
    for node in g:
        if nx.has_eulerian_path(G, source=node):
            raw_path = list(nx.eulerian_path(G, source=node))
            path = shorten_path(raw_path)
            break
    return path


def graph2path_test(graph: Data) -> List[Tuple[int]]:
    G = to_networkx(graph, to_undirected="upper").to_undirected()
    # 1. Eulerize the graph if it is not
    G = connect_graph(G)
    if not nx.is_eulerian(G):
        G = nx.eulerize(G)
    # 2. loop through nodes, and get one euler path if exists
    g = list(G.nodes())
    random.shuffle(g)
    for node in g:
        if nx.has_eulerian_path(G, source=node):
            raw_path = list(nx.eulerian_path(G, source=node))
            path = shorten_path(raw_path)
            break
    # ls = list(range(graph.num_nodes))
    # path = list(zip(ls[:-1], ls[1:]))
    return path


def get_precalculated_path(graph: Data) -> List[Tuple[int]]:
    paths = (
        torch.sparse_coo_tensor(
            indices=graph.paths_ind.T,
            values=graph.paths_val,
            size=graph.paths_shape.tolist(),
        ).to_dense()
        - 1
    )
    idx = random.choice(range(paths.shape[0]))
    ls_nodes = [node for node in paths[idx].tolist() if node != -1]
    path = list(zip(ls_nodes[:-1], ls_nodes[1:]))
    return path


def get_paths(graph: Data, form: str = "pair") -> List[Union[Tuple[int], int]]:
    # For preprocess small-medium graphs and store the paths
    assert form in {"pair", "singular"}
    G = to_networkx(graph, to_undirected="upper").to_undirected()
    # 1. Eulerize the graph if it is not
    # G = connect_graph(G)  # if the graph is disconnected, we prefer to generate the path dynamically instead of save pre-calculated paths
    if not nx.is_eulerian(G):
        G = nx.eulerize(G)
    # 2. loop through nodes, and get all euler paths if exists
    ls_paths = []
    for node in G.nodes():
        if nx.has_eulerian_path(G, source=node):
            raw_path = list(nx.eulerian_path(G, source=node))
            path = shorten_path(raw_path)
            path = (
                [src for src, tgt in path] + [path[-1][-1]]
                if form == "singular"
                else path
            )
            ls_paths.append(path)
    return ls_paths


def add_paths(graph: Data) -> Data:
    ls_paths = get_paths(
        graph, "singular"
    )  # TODO: whether to deduplicate paths by re-indexing the nodes? how to dedup if got node/edge attrs
    res = itertools.zip_longest(*ls_paths, fillvalue=-1)
    paths = torch.tensor(list(res), dtype=torch.int64).T
    # Turn into sparse format so that it can be stored by torch_geometric
    sparsed_paths = (paths + 1).to_sparse()
    graph.paths_ind = sparsed_paths.indices().T
    graph.paths_val = sparsed_paths.values()
    graph.paths_shape = torch.tensor(sparsed_paths.shape)
    return graph


def _reindex_node_pairs(path: List[Tuple[int]]):
    """
    Re-index the (src, tgt) node-pairs in each path
    :param path:
    :return:
    """
    # 1. initialize the mapping
    idx = 1  # 1st element starts from 1 instead of 0
    dict_map = {}
    for src, tgt in path:
        if dict_map.get(src, None) is None:
            dict_map[src] = idx
            idx += 1
    if (
        dict_map.get(tgt, None) is None
    ):  # for semi-euler path OR shortened euler path, which does not go back to origin
        dict_map[tgt] = idx
    # 2. apply the mapping
    new_path = [(dict_map[src], dict_map[tgt]) for src, tgt in path]
    return tuple(new_path)


def _reindex_node_singulars(path: List[int]):
    """
    Re-index the node singulars in each path
    :param path:
    :return:
    """
    # 1. initialize the mapping
    idx = 1  # 1st element starts from 1 instead of 0
    dict_map = {}
    for node in path:
        if dict_map.get(node, None) is None:
            dict_map[node] = idx
            idx += 1
    # 2. apply the mapping
    new_path = [dict_map[node] for node in path]
    return tuple(new_path)


@_inputs_deco("pretrain")
def prepare_inputs_for_pretrain(in_dict, **kwargs):
    return in_dict


@_inputs_deco("graph-dh")
def prepare_inputs_for_graph_lvl_double_head_task(
    in_dict: Dict[str, List[int]],
    *,
    graph: Data,
    eos_token_id: int,
    gsum_token_id: int,
    tgt_pos: torch.Tensor,
    gtokenizer,
    **kwargs
):
    """
    inputs Graph-level DoubleHead tasks:
        task1 -> unidirectional NTP task
        task2 -> bidirectional regression task, e.g., predict 3d coordinates (noises)
    :param in_dict:
    :param graph:
    :param eos_token_id:
    :param gsum_token_id:
    :param tgt_pos: [num_nodes, 3]
    :param gtokenizer:
    :param kwargs:
    :return:
    """
    assert gsum_token_id is not None
    num_nodes = graph.x.shape[0] if torch.abs(tgt_pos).sum() > 1e-8 else 0
    assert (
        num_nodes <= gtokenizer.config["structure"]["node"]["scope_base"]
    ), "NOT Implemented"
    assert (tgt_pos.shape[0] == num_nodes) or (num_nodes == 0)
    # 1. add node-id as extended tokens for 3d position regression task
    ls_node_tokens = [str(x) for x in range(num_nodes)]
    ls_node_tokens_id = [gtokenizer.vocab_map[x] for x in ls_node_tokens]
    # ls_extend_tokens = [eos_token_id, gsum_token_id]
    ls_extend_tokens = [gsum_token_id] + ls_node_tokens_id
    # 2. based on the extended tokens-id, reformat the input dict elements
    len_extended_tokens = len(ls_extend_tokens)
    in_dict["input_ids"].extend(ls_extend_tokens)
    in_dict["position_ids"].extend(
        list(
            range(
                len(in_dict["position_ids"]),
                len(in_dict["position_ids"]) + len_extended_tokens,
            )
        )
    )
    in_dict["labels"].extend([-100] * len_extended_tokens)
    in_dict["attention_mask"].extend([1] * len_extended_tokens)
    in_dict["graph_labels"] = torch.squeeze(graph.y).tolist()
    # 2.1 attention_mask_bi for the 3d position regression task
    seq = len(in_dict["attention_mask"])
    in_dict["attention_mask_bi"] = [0 if i < seq - num_nodes else 1 for i in range(seq)]
    # 2.2 pad the 3d position tensor: [N,3] -> [N+1+euler-seq-len,3]
    if torch.abs(tgt_pos).sum() > 1e-8:
        pos_pad = torch.zeros(seq - num_nodes, 3, dtype=torch.float32)
        in_dict["pos"] = torch.cat([pos_pad, tgt_pos], dim=0)
    else:
        in_dict["pos"] = torch.zeros(seq, 3, dtype=torch.float32)
    return in_dict


@_inputs_deco("graph")
def prepare_inputs_for_graph_lvl_task(
    in_dict: Dict[str, List[int]],
    *,
    graph: Data,
    eos_token_id: int,
    gsum_token_id: int,
    **kwargs
):
    assert gsum_token_id is not None
    # ls_extend_tokens = [eos_token_id, gsum_token_id]
    ls_extend_tokens = [gsum_token_id]  # to be compatible with lf's best result setting
    len_extended_tokens = len(ls_extend_tokens)
    in_dict["input_ids"].extend(ls_extend_tokens)
    in_dict["position_ids"].extend(
        list(
            range(
                len(in_dict["position_ids"]),
                len(in_dict["position_ids"]) + len_extended_tokens,
            )
        )
    )
    in_dict["labels"].extend([-100] * len_extended_tokens)
    in_dict["attention_mask"].extend([1] * len_extended_tokens)
    in_dict["graph_labels"] = torch.squeeze(graph.y).tolist()
    return in_dict


@_inputs_deco("edge")
def prepare_inputs_for_edge_lvl_task(
    in_dict: Dict[str, List[int]],
    *,
    graph: Data,
    eos_token_id: int,
    tgt_edge_src_token_id: Union[int, Tuple, List],
    tgt_edge_dst_token_id: Union[int, Tuple, List],
    **kwargs
):
    ls_src_dst = [tgt_edge_src_token_id, tgt_edge_dst_token_id]
    random.shuffle(ls_src_dst)
    if isinstance(tgt_edge_dst_token_id, Tuple) or isinstance(
        tgt_edge_dst_token_id, List
    ):
        ls_src_dst = [item for row in ls_src_dst for item in row]
    extended_token_len = len(ls_src_dst) + 1  # +1 for eos token
    in_dict["idx"] = (
        graph.seed_node.tolist() if hasattr(graph, "seed_node") else ls_src_dst
    )
    in_dict["input_ids"].extend([eos_token_id] + ls_src_dst)
    in_dict["position_ids"].extend(
        list(
            range(
                len(in_dict["position_ids"]),
                len(in_dict["position_ids"]) + extended_token_len,
            )
        )
    )
    in_dict["labels"].extend([-100] * extended_token_len)
    in_dict["attention_mask"].extend([1] * extended_token_len)
    in_dict["edge_labels"] = graph.y.item()
    return in_dict


@_inputs_deco("node")
def prepare_inputs_for_node_lvl_task(
    in_dict: Dict[str, List[int]],
    *,
    graph: Data,
    eos_token_id: int,
    tgt_node_token_id: Union[int, Tuple],
    **kwargs
):
    if isinstance(tgt_node_token_id, int):
        ls_token_ids = [tgt_node_token_id]
    else:
        ls_token_ids = list(
            tgt_node_token_id
        )  # for node identity encoding with multiple tokens
    extended_token_len = len(ls_token_ids) + 1  # +1 for eos token
    in_dict["idx"] = ls_token_ids
    in_dict["input_ids"].extend([eos_token_id] + ls_token_ids)
    in_dict["position_ids"].extend(
        [
            len(in_dict["position_ids"]),
            len(in_dict["position_ids"]) + extended_token_len,
        ]
    )
    in_dict["labels"].extend([-100] * extended_token_len)
    in_dict["attention_mask"].extend([1] * extended_token_len)
    assert graph.num_nodes == graph.y.shape[0]
    in_dict["node_labels"] = graph.y[graph.root_n_id].tolist()
    return in_dict
