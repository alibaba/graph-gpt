"""Supervised task preparation strategies."""

import torch
from .base import TaskPreparationStrategy


class GraphLevelStrategy(TaskPreparationStrategy):
    """Graph-level classification/regression task preparation."""

    def prepare(self, in_dict, token_res, graph, gtokenizer):
        """Prepare inputs for graph-level tasks."""
        ls_raw_node_idx = token_res.ls_raw_node_idx

        # Add graph labels
        in_dict["graph_labels"] = torch.squeeze(graph.y).tolist()

        # Handle embeddings extension
        len_extended_tokens = 0
        if "embed" in in_dict:
            import numpy as np

            dim = len(in_dict["embed"][0])
            extended_embed = np.zeros(
                (len_extended_tokens, dim), dtype=np.float32
            ).tolist()
            in_dict["embed"].extend(extended_embed)
            assert len(in_dict["embed"]) == len(in_dict["input_ids"])

        # Attach node mask if available
        if ls_raw_node_idx is not None:
            in_dict = self._attach_node_mask(
                in_dict, ls_raw_node_idx, len_extended_tokens
            )

        in_dict["split_lens"] = [len(in_dict["input_ids"])]
        in_dict["attn_modes"] = ["full"]
        return in_dict

    def _attach_node_mask(self, in_dict, ls_raw_node_idx, len_extended_tokens):
        """Attach node mask to inputs."""
        from ...masking import get_mask_of_raw_seq
        import numpy as np

        ls_raw_node_idx = list(ls_raw_node_idx)
        ls_raw_node_idx.extend([-1] * len_extended_tokens)
        node_idx = np.array(ls_raw_node_idx) + 1
        node_idx_clip = np.clip(node_idx, 0, 4)
        node_mask = get_mask_of_raw_seq(node_idx, mask_type="random")
        node_mask = node_mask * (node_idx > 0)
        edge_seq = list(zip([0] + node_idx.tolist()[:-1], node_idx.tolist()))
        edge_mask = get_mask_of_raw_seq(edge_seq, mask_type="random")
        edge_mask = edge_mask * (np.array(edge_seq) > 0).all(axis=-1)
        node_type = np.vstack([node_idx_clip, node_mask, node_idx, edge_mask]).T
        in_dict["input_ids"] = np.hstack(
            [np.array(in_dict["input_ids"]), node_type]
        ).tolist()
        return in_dict


class EdgeLevelStrategy(TaskPreparationStrategy):
    """Edge-level link prediction task preparation."""

    def prepare(self, in_dict, token_res, graph, gtokenizer):
        """Prepare inputs for edge-level tasks."""
        tgt_edge_src_token_id = self._map_tokens_to_ids(
            token_res.tgt_edge_src_token, gtokenizer.vocab_map
        )
        tgt_edge_dst_token_id = self._map_tokens_to_ids(
            token_res.tgt_edge_dst_token, gtokenizer.vocab_map
        )
        tgt_edge_attr_token_id = self._map_tokens_to_ids(
            token_res.tgt_edge_attr_token, gtokenizer.vocab_map
        )

        # Prepare source/dst tokens
        ls_src_dst = [tgt_edge_src_token_id, tgt_edge_dst_token_id]
        if not tgt_edge_attr_token_id:
            import random

            random.shuffle(ls_src_dst)

        # Flatten if needed
        if isinstance(tgt_edge_dst_token_id, (tuple, list)):
            ls_src_dst = [item for row in ls_src_dst for item in row]

        raw_ls_extend_tokens = list(ls_src_dst)
        ls_extend_tokens = list(ls_src_dst)
        ls_extend_emb = []

        # Handle 2D (stacked) tokens
        if isinstance(in_dict["input_ids"][0], list):
            dict_mapping = {x[0]: x for x in in_dict["input_ids"]}
            ls_extend_tokens = [list(dict_mapping[x]) for x in raw_ls_extend_tokens]

            edge_dim = gtokenizer.config["semantics"]["edge"]["dim"]
            if edge_dim > 0:
                assert len(ls_extend_tokens) == 2
                default_edge_attr_id = gtokenizer.get_default_edge_attr_id()
                assert len(default_edge_attr_id) == edge_dim
                ls_extend_tokens[0] = ls_extend_tokens[0][:-edge_dim] + list(
                    default_edge_attr_id
                )
                if tgt_edge_attr_token_id:
                    assert len(tgt_edge_attr_token_id) == edge_dim
                    ls_extend_tokens[1] = ls_extend_tokens[1][:-edge_dim] + list(
                        tgt_edge_attr_token_id
                    )
                else:
                    ls_extend_tokens[1] = ls_extend_tokens[1][:-edge_dim] + list(
                        default_edge_attr_id
                    )

            # Handle embeddings
            if "embed" in in_dict:
                assert len(in_dict["input_ids"]) == len(in_dict["embed"])
                dict_emb_mapping = {
                    x[0]: y for x, y in zip(in_dict["input_ids"], in_dict["embed"])
                }
                ls_extend_emb = [
                    list(dict_emb_mapping[x]) for x in raw_ls_extend_tokens
                ]

        # Extend input dict
        in_dict = self._extend_input_dict(in_dict, ls_extend_tokens)

        # Add metadata
        in_dict["idx"] = (
            graph.seed_node.tolist() if hasattr(graph, "seed_node") else ls_src_dst
        )
        in_dict["edge_labels"] = graph.y.item()

        if "embed" in in_dict:
            in_dict["embed"].extend(ls_extend_emb)
            assert len(in_dict["input_ids"]) == len(in_dict["embed"])

        if hasattr(graph, "wgt"):
            in_dict["wgt"] = graph.wgt.item()

        in_dict["split_lens"] = [len(in_dict["input_ids"])]
        in_dict["attn_modes"] = ["full"]
        return in_dict


class NodeLevelStrategy(TaskPreparationStrategy):
    """Node-level classification/regression task preparation."""

    def prepare(self, in_dict, token_res, graph, gtokenizer):
        """Prepare inputs for node-level tasks."""
        tgt_node_token_id = self._map_tokens_to_ids(
            token_res.tgt_node_token, gtokenizer.vocab_map
        )

        # Convert to list
        if isinstance(tgt_node_token_id, int):
            ls_token_ids = [tgt_node_token_id]
        else:
            ls_token_ids = list(tgt_node_token_id)

        raw_ls_extend_tokens = list(ls_token_ids)
        ls_extend_tokens = list(ls_token_ids)
        ls_extend_emb = []

        # Handle 2D (stacked) tokens
        if isinstance(in_dict["input_ids"][0], list):
            dict_mapping = {x[0]: x for x in in_dict["input_ids"]}
            ls_extend_tokens = [list(dict_mapping[x]) for x in raw_ls_extend_tokens]

            edge_dim = gtokenizer.config["semantics"]["edge"]["dim"]
            if edge_dim > 0:
                assert len(ls_extend_tokens) == 1
                default_edge_attr_id = gtokenizer.get_default_edge_attr_id()
                assert len(default_edge_attr_id) == edge_dim
                ls_extend_tokens[0] = ls_extend_tokens[0][:-edge_dim] + list(
                    default_edge_attr_id
                )

            # Handle embeddings
            if "embed" in in_dict:
                assert len(in_dict["input_ids"]) == len(in_dict["embed"])
                dict_emb_mapping = {
                    x[0]: y for x, y in zip(in_dict["input_ids"], in_dict["embed"])
                }
                ls_extend_emb = [
                    list(dict_emb_mapping[x]) for x in raw_ls_extend_tokens
                ]

        # Extend input dict
        in_dict = self._extend_input_dict(in_dict, ls_extend_tokens)

        # Add metadata
        in_dict["idx"] = ls_token_ids
        assert graph.num_nodes == graph.y.shape[0]
        in_dict["node_labels"] = graph.y[graph.root_n_id].tolist()

        if "embed" in in_dict:
            in_dict["embed"].extend(ls_extend_emb)
            assert len(in_dict["input_ids"]) == len(in_dict["embed"])

        if hasattr(graph, "wgt"):
            in_dict["wgt"] = graph.wgt.item()

        in_dict["split_lens"] = [len(in_dict["input_ids"])]
        in_dict["attn_modes"] = ["full"]
        return in_dict


class NodeV2Strategy(TaskPreparationStrategy):
    """NodeV2 token-level node classification preparation."""

    def prepare(self, in_dict, token_res, graph, gtokenizer):
        """Prepare inputs for NodeV2 tasks."""
        tgt_node_token_id = self._map_tokens_to_ids(
            token_res.tgt_node_token, gtokenizer.vocab_map
        )

        # Get node labels
        if (
            hasattr(graph, "y")
            and (graph.y is not None)
            and (graph.y.shape[0] == graph.num_nodes)
        ):
            nodev2_labels = graph.y[:, 0].tolist()
        else:
            nodev2_labels = [-100] * graph.x.shape[0]

        assert len(tgt_node_token_id) == len(nodev2_labels)
        mapping = dict(zip(tgt_node_token_id, nodev2_labels))
        mapping2raw_node_idx = dict(
            zip(tgt_node_token_id, list(range(len(nodev2_labels))))
        )

        # Create nodev2 labels and raw node indices
        if isinstance(in_dict["input_ids"][0], int):
            in_dict["nodev2_labels"] = [
                mapping.pop(ele, -100) for ele in in_dict["input_ids"]
            ]
            in_dict["raw_node_idx"] = [
                mapping2raw_node_idx.pop(ele, -100) for ele in in_dict["input_ids"]
            ]
        else:
            in_dict["nodev2_labels"] = [
                mapping.pop(ele[0], -100) for ele in in_dict["input_ids"]
            ]
            in_dict["raw_node_idx"] = [
                mapping2raw_node_idx.pop(ele[0], -100) for ele in in_dict["input_ids"]
            ]

        # Handle token_ce_intra loss type
        loss_type = gtokenizer.kwargs.get("loss_type", "token_ce")
        num_labels = gtokenizer.kwargs.get("num_labels", 10)
        permute_label = gtokenizer.kwargs.get("permute_label", True)

        if loss_type == "token_ce_intra":
            reserved_semantics_tokens = gtokenizer.config["semantics"]["common"].get(
                "reserved_token", []
            )
            assert len(reserved_semantics_tokens) >= num_labels
            if permute_label:
                import random

                random.shuffle(reserved_semantics_tokens)
            in_dict["cls_idx"] = [len(in_dict["input_ids"])]
            ls_extend_tokens = [
                gtokenizer.vocab_map[x] for x in reserved_semantics_tokens
            ]
            in_dict = self._extend_input_dict(
                in_dict,
                ls_extend_tokens,
                keys=("nodev2_labels", "raw_node_idx"),
                vals=(-100, -100),
            )

        in_dict["split_lens"] = [len(in_dict["input_ids"])]
        in_dict["attn_modes"] = ["full"]
        return in_dict
