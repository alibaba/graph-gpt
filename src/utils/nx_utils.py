import sys
import random
import networkx as nx
import itertools
from typing import List, Tuple, Dict, Union, Iterable
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from . import control_flow

_nx = control_flow.Register()
get_nx_struct = _nx.build  # return func results


def understand_structure(
    graph: Data,
    tokenization_config: Dict,
    node_structure_mapping: Dict,
    edge_structure_mapping: Dict,
    gtokenizer=None,
):
    ls = []
    ls_labels = []
    nx_conf = tokenization_config["structure"].get("nx", {})
    if nx_conf.get("enable", False):
        G = to_networkx(graph, to_undirected="upper").to_undirected()
        for func in nx_conf["func"]:
            if func["valid"]:
                tmp_ls, tmp_ls_labels = get_nx_struct(
                    func["name"],
                    G,
                    node_structure_mapping=node_structure_mapping,
                    edge_structure_mapping=edge_structure_mapping,
                    config=tokenization_config,
                    graph=graph,
                    gtokenizer=gtokenizer,
                )
                ls.append(tmp_ls)
                ls_labels.append(tmp_ls_labels)
        if len(ls) > 1:
            idx_tmp = list(range(len(ls)))
            random.shuffle(idx_tmp)
            ls = [ls[idx] for idx in idx_tmp]
            ls_labels = [ls_labels[idx] for idx in idx_tmp]

        ls = _flatten_list(ls)
        ls_labels = _flatten_list(ls_labels)
    return ls, ls_labels


@_nx("degree")
def _obtain_node_degree(
    G: nx.Graph, *, node_structure_mapping: Dict, config: Dict, gtokenizer, **kwargs
):
    reserved_token_id = 0
    idx = random.choice(list(G.nodes()))
    num = G.degree[idx]

    func_tokens = [config["structure"]["common"]["reserved_token"][reserved_token_id]]
    tgt_node_tokens = list(node_structure_mapping[idx])
    num_tokens = [f"<{ele}>" for ele in list(str(num))]
    return func_tokens + tgt_node_tokens + num_tokens


@_nx("triangles")
def _obtain_graph_triangles(
    G: nx.Graph, *, node_structure_mapping: Dict, config: Dict, gtokenizer, **kwargs
):
    reserved_token_id = 1
    idx = random.choice(list(G.nodes()))
    num = nx.triangles(G, idx)

    func_tokens = [config["structure"]["common"]["reserved_token"][reserved_token_id]]
    tgt_node_tokens = list(node_structure_mapping[idx])
    num_tokens = [f"<{ele}>" for ele in list(str(num))]
    return func_tokens + tgt_node_tokens + num_tokens


@_nx("shortest_path")
def _obtain_edge_shortest_path(
    G: nx.Graph, *, node_structure_mapping: Dict, config: Dict, gtokenizer, **kwargs
):
    reserved_token_id = 2
    if len(G.nodes()) > 2:
        src, tgt = random.sample(list(G.nodes()), 2)
        try:
            ls_nodes = nx.shortest_path(G, source=src, target=tgt)
        except nx.NetworkXNoPath:
            ls_nodes = []

        func_tokens = [
            config["structure"]["common"]["reserved_token"][reserved_token_id]
        ]
        tgt_node_tokens = [node_structure_mapping[node] for node in [src, tgt]]
        seq_tokens = [node_structure_mapping[node] for node in ls_nodes]
        return func_tokens + _flatten_list(tgt_node_tokens) + _flatten_list(seq_tokens)
    else:
        return []


@_nx("shortest_path_length")
def _obtain_edge_shortest_path_length(
    G: nx.Graph, *, node_structure_mapping: Dict, config: Dict, gtokenizer, **kwargs
):
    reserved_token_id = 3
    if len(G.nodes()) > 2:
        src, tgt = random.sample(list(G.nodes()), 2)
        try:
            num = nx.shortest_path_length(G, source=src, target=tgt)
        except nx.NetworkXNoPath:
            num = -1

        func_tokens = [
            config["structure"]["common"]["reserved_token"][reserved_token_id]
        ]
        tgt_node_tokens = [node_structure_mapping[node] for node in [src, tgt]]
        num_tokens = [f"<{ele}>" for ele in list(str(num))]
        return func_tokens + _flatten_list(tgt_node_tokens) + num_tokens
    else:
        return []


@_nx("eulerian_path")
def _obtain_eulerian_path(
    G: nx.Graph,
    *,
    node_structure_mapping: Dict,
    config: Dict,
    graph: Data,
    gtokenizer,
    **kwargs,
):
    reserved_token_id = 4
    graph, permu = permute_nodes(graph)
    path, old_node = _get_new_eulerian_path_v2(graph, permu, node_structure_mapping)
    tgt_node_tokens = [node_structure_mapping[old_node]]

    func_tokens = [config["structure"]["common"]["reserved_token"][reserved_token_id]]
    local_node_structure_mapping = get_structure_raw_node2idx_mapping(
        path,
        config["structure"]["node"]["scope_base"],
        config["structure"]["node"]["node_scope"],
        config["structure"]["node"].get("cyclic", False),
    )
    local_edge_structure_mapping = get_structure_raw_edge2type_mapping(path, graph)
    raw_seq = get_raw_seq_from_path(path)
    mask = [False] * len(raw_seq)
    ls_tokens, _, _ = decorate_node_edge_graph_with_mask(
        None,
        raw_seq,
        mask,
        local_node_structure_mapping,
        local_edge_structure_mapping,
        {},
        {},
        {"discrete": {}},
        attr_shuffle=False,
    )
    dict_edge = config["structure"]["edge"]
    if dict_edge.get("remove_edge_type_token", False):
        edge_types = {dict_edge["bi_token"]}
        ls_tokens = [token for token in ls_tokens if token not in edge_types]
    # p = [src for src, tgt in path] + [path[-1][-1]]
    # tgt_node_tokens = [node_structure_mapping[node] for node in p]
    prefix_tokens = func_tokens + _flatten_list(tgt_node_tokens)
    ls_tokens = prefix_tokens + ls_tokens
    ls_labels = get_labels_from_input_tokens(
        ls_tokens, gtokenizer, skipped=len(prefix_tokens)
    )
    return ls_tokens, ls_labels


def _get_new_eulerian_path_v1(graph, permu, node_structure_mapping):
    G = to_networkx(graph, to_undirected="upper").to_undirected()
    G = connect_graph(G)
    if not nx.is_eulerian(G):
        G = nx.eulerize(G)
    path = []  # in case of single node graph
    g = list(G.nodes())
    random.shuffle(g)
    for old_node in g:
        if node_structure_mapping[old_node] != ("0",):
            new_node = permu[old_node].item()
            raw_path = list(_customized_eulerian_path(G, source=new_node))
            path = shorten_path(raw_path)
            break
    return path, old_node


def _get_new_eulerian_path_v2(graph, permu, node_structure_mapping):
    path = graph2path_v2(graph)
    if len(path) == 0:
        assert len(node_structure_mapping) == 1
        start_node = 0
    else:
        start_node = path[0][0]
    for old_node in range(len(node_structure_mapping)):
        if start_node == permu[old_node].item():
            break
    return path, old_node


def _customized_eulerian_path(G, source):
    # To enhance randomization of eulerian path, thus as a kind of data augmentation
    if random.random() < 0.5:
        return nx.eulerian_path(G, source=source)
    else:
        return nx.eulerian_circuit(G, source=source)


def _flatten_list(ls):
    if isinstance(ls[0], str):
        return ls
    elif isinstance(ls[0], Iterable):
        return [ele for sub_ls in ls for ele in sub_ls]
    else:
        raise ValueError(
            f"ls' element must be str or Iterable, but yours {ls[0]} is {type(ls[0])}"
        )


def _rebase_idx(idx: int, base: int):
    if base == 0:
        return f"{idx}"
    assert idx < base * base
    idx_1 = idx // base
    idx_2 = idx - idx_1 * base
    rebased_idx = (f"{idx_1}*{base}", str(idx_2)) if idx_1 > 0 else (str(idx_2),)
    return rebased_idx


def get_structure_raw_node2idx_mapping(
    path: List[Tuple[int, int]], scope_base: int, scope: int, mapping_type: int = 0
):
    # mapping_type: 0/1/2 -> normal/cyclic/random
    mapping_type = int(mapping_type)
    # refer: https://stackoverflow.com/a/17016257/4437068
    assert (sys.version_info.major == 3) and (sys.version_info.minor >= 7)
    # `random.randint` Return random integer in range [a, b], including both end points.
    start_idx = random.randint(0, scope - 1) if mapping_type > 0 else 0
    if path:
        path_s = [src for src, tgt in path]
        path_s.append(path[-1][-1])
        uniques = list(dict.fromkeys(path_s))
        if mapping_type == 2:
            rnd_idx = random.sample(range(scope), k=len(uniques))
            dict_map = {
                old_idx: _rebase_idx(idx, scope_base)
                for idx, old_idx in zip(rnd_idx, uniques)
            }
        else:
            dict_map = {
                old_idx: _rebase_idx(idx % scope, scope_base)
                for idx, old_idx in enumerate(uniques, start=start_idx)
            }
    else:  # in case `path=[]` when graph has ONLY 1 node
        dict_map = {0: _rebase_idx(start_idx % scope, scope_base)}
    return dict_map


def get_structure_raw_edge2type_mapping(path: List[Tuple[int, int]], data: Data):
    # map the edge to its type
    dict_map = {
        (src, tgt): get_edge_type(data.edge_index, src, tgt) for src, tgt in path
    }
    return dict_map


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


def graph2path(graph: Data, prioritize: bool = False) -> List[Tuple[int, int]]:
    return graph2path_v2(graph)
    # return graph2path_v1(graph, prioritize)


def graph2path_v1(graph: Data, prioritize: bool = False) -> List[Tuple[int, int]]:
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
        # if nx.has_eulerian_path(G, source=node):
        raw_path = list(_customized_eulerian_path(G, source=node))
        path = shorten_path(raw_path)
        break
    # comment above and use below to be compatible with xin-shuai's data processing
    # below will easily cause overfitting after several epochs since the path seq is fixed!
    # raw_path = list(nx.eulerian_circuit(G, source=0))
    # path = shorten_path(raw_path)
    return path


def graph2path_v2(graph: Data) -> List[Tuple[int, int]]:
    G = to_networkx(graph, to_undirected="upper").to_undirected()
    # 1. create list of subgraphs
    if not nx.is_connected(G):
        S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    else:
        S = [G]
    # 2. find eulerian paths in each subgraph, and then concat sub-paths
    random.shuffle(S)
    s = S[0]
    path = connected_graph2path(s)
    prev_connect_node = list(s.nodes)[0] if len(path) == 0 else path[-1][-1]
    for s in S[1:]:
        spath = connected_graph2path(s)
        if len(spath) == 0:  # single node
            curr_connect_node = list(s.nodes)[0]
        else:
            curr_connect_node = spath[0][0]
        jump_edge = (prev_connect_node, curr_connect_node)
        path.append(jump_edge)
        path.extend(spath)
        prev_connect_node = path[-1][-1]
    return path


def connected_graph2path(G) -> List[Tuple[int, int]]:
    if len(G.nodes) == 1:
        path = []
    else:
        if not nx.is_eulerian(G):
            G = nx.eulerize(G)
        node = random.choice(list(G.nodes()))
        raw_path = list(_customized_eulerian_path(G, source=node))
        path = shorten_path(raw_path)
    return path


def get_raw_seq_from_path(path):
    # raw_seq:: [<node>, <edge>, <node>, <edge>, ...]
    # <node> in the format of int, e.g., 3
    # <edge> in the format of tuple of int, e.g., (3, 0)
    raw_seq = []
    if path:
        for src, tgt in path:
            raw_seq.append(src)
            raw_seq.append((src, tgt))
        raw_seq.append(tgt)
    else:  # in case `path=[]` when graph has ONLY 1 node
        raw_seq.append(0)
    return raw_seq


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
        raw_path = list(_customized_eulerian_path(G, source=node))
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
        raw_path = list(_customized_eulerian_path(G, source=node))
        path = shorten_path(raw_path)
        path = (
            [src for src, tgt in path] + [path[-1][-1]] if form == "singular" else path
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


def _unfold_ls_of_ls(ls, shuffle: bool = False):
    random.shuffle(ls) if shuffle else None
    if isinstance(ls[0], list) or isinstance(ls[0], tuple):
        ls = [item for row in ls for item in row]
    return ls


def decorate_node_edge_graph_with_mask(
    gtokenizer,  # GSTTokenizer
    raw_seq,
    mask,
    node_structure_mapping,
    edge_structure_mapping,
    node_semantics_mapping,
    edge_semantics_mapping,
    graph_semantics_mapping,
    attr_shuffle: bool = False,
):
    ls_tokens = []  # For next-token-prediction
    ls_node_regression_labels = []  # Groundtruth for predict continuous node attr
    ls_edge_regression_labels = []  # Groundtruth for predict continuous edge attr
    for i, (raw_token, is_deco) in enumerate(zip(raw_seq, mask)):
        if i % 2 == 0:  # deco node
            node_id = node_structure_mapping[raw_token]
            # node_id will be List if it is represented by several ids, e.g., global+local-ids
            ls_tokens.extend(node_id) if isinstance(node_id, Tuple) or isinstance(
                node_id, List
            ) else ls_tokens.append(node_id)
            if is_deco:
                node_attr = node_semantics_mapping["discrete"].get(raw_token, None)
                if node_attr:
                    ls_tokens.extend(_unfold_ls_of_ls(node_attr, attr_shuffle))
        else:  # deco edge
            edge_type = edge_structure_mapping[raw_token]
            ls_tokens.append(edge_type)
            if is_deco:
                edge_attr = edge_semantics_mapping["discrete"].get(raw_token, None)
                if edge_attr:
                    ls_tokens.extend(_unfold_ls_of_ls(edge_attr, attr_shuffle))
    # deco graph
    graph_attr = graph_semantics_mapping["discrete"].get(0, [])
    ls_tokens.extend(
        [gtokenizer.get_eos_token()] + _unfold_ls_of_ls(graph_attr, False)
    ) if len(graph_attr) > 0 else None
    return ls_tokens, ls_node_regression_labels, ls_edge_regression_labels


def permute_nodes(graph, g=None):
    if hasattr(graph, "num_nodes"):
        num_nodes = graph.num_nodes
    elif isinstance(graph.x, torch.Tensor):
        num_nodes = graph.x.shape[0]
    else:
        num_nodes = graph.edge_index.max().item() + 1
    permu = torch.randperm(num_nodes, generator=g)
    new_graph = graph.clone()
    new_graph.edge_index = permu[graph.edge_index]
    new_graph.num_nodes = num_nodes

    inv_permu = torch.argsort(permu)
    for k, v in new_graph:
        if k in ["edge_index", "adj_t", "num_nodes", "batch", "ptr"]:
            continue
        if isinstance(v, Tensor) and v.size(0) == new_graph.num_nodes:
            new_graph[k] = graph[k][inv_permu]
    return new_graph, permu


def get_labels_from_input_tokens(ls_tokens, gtokenizer, skipped=0):
    mapping_type = int(gtokenizer.config["structure"]["node"].get("cyclic", False))
    if len(ls_tokens) > 0:
        ls_labels = ls_tokens[1:] + [gtokenizer.get_eos_token()]
        for i, token in enumerate(ls_labels):
            if (
                token not in set(ls_tokens[skipped:i])
                and token in gtokenizer.get_node_idx_tokens()
                and mapping_type == 2
            ):
                ls_labels[i] = gtokenizer.get_new_node_token()
            if i < skipped:
                ls_labels[i] = gtokenizer.get_label_pad_token()
        return ls_labels
    else:
        return []
