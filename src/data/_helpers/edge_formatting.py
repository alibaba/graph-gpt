import torch


def _get_edge_neg(source_node, target_node_neg):
    if len(source_node.shape) == 2:
        # invoke itself
        edge_neg = _get_edge_neg(target_node_neg, source_node)
        edge_neg = edge_neg[:, [1, 0]].clone()
    else:
        assert len(source_node.shape) == 1
        assert len(target_node_neg.shape) == 2
        assert source_node.shape[0] == target_node_neg.shape[0]
        num_negs = target_node_neg.shape[1]
        # [N_p] -> [N_p,1] -> [N_p, num_negs]
        source_node = source_node.reshape((-1, 1)).expand(
            source_node.shape[0], num_negs
        )
        assert source_node.shape == target_node_neg.shape
        edge_neg = torch.hstack(
            [source_node.reshape((-1, 1)), target_node_neg.reshape((-1, 1))]
        ).clone()
    return edge_neg


def _get_fixed_sampled_sorted_idx(scope, cnt_idx, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(scope, generator=g)[:cnt_idx]
    indices, _ = torch.sort(indices)
    return indices


def _get_reformatted_data_of_citation2(dict_, idx_sel):
    edge = torch.vstack(
        [
            dict_["source_node"][idx_sel],
            dict_["target_node"][idx_sel],
        ]
    ).T.clone()
    edge_neg = _get_edge_neg(
        dict_["source_node"][idx_sel],
        dict_["target_node_neg"][idx_sel],
    )
    pos_edge_attr = torch.ones((edge.shape[0], 1), dtype=torch.int64)
    neg_edge_attr = torch.ones((edge_neg.shape[0], 1), dtype=torch.int64)
    return {
        "edge": edge,
        "edge_neg": edge_neg,
        "pos_edge_attr": pos_edge_attr,
        "neg_edge_attr": neg_edge_attr,
    }


def _get_reformatted_data_of_wikikg2(dict_, idx_sel):
    edge = torch.vstack(
        [
            dict_["head"][idx_sel],
            dict_["tail"][idx_sel],
        ]
    ).T.clone()
    rel = dict_["relation"][idx_sel]
    pos_edge_attr = torch.vstack([torch.ones_like(rel), rel]).T.clone()

    edge_neg1 = _get_edge_neg(
        dict_["head"][idx_sel],
        dict_["head_neg"][idx_sel],
    )
    edge_neg2 = _get_edge_neg(
        dict_["tail_neg"][idx_sel],
        dict_["tail"][idx_sel],
    )
    edge_neg = torch.hstack([edge_neg1, edge_neg2]).reshape((-1, 2)).clone()

    neg_rel = rel.reshape((-1, 1)).expand(rel.shape[0], 1000).reshape((-1, 1))
    neg_edge_attr = torch.hstack([torch.ones_like(neg_rel), neg_rel]).clone()

    return {
        "edge": edge,
        "edge_neg": edge_neg,
        "pos_edge_attr": pos_edge_attr,
        "neg_edge_attr": neg_edge_attr,
    }
