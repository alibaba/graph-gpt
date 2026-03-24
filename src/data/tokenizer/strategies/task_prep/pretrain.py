"""Pre-training task preparation strategies."""

import numpy as np
from .base import TaskPreparationStrategy


class PretrainMLMStrategy(TaskPreparationStrategy):
    """Masked Language Model pre-training preparation."""

    def prepare(self, in_dict, token_res, graph, gtokenizer):
        """Prepare inputs for MLM pre-training."""
        from ...masking import (
            _get_mask_ratio,
            _mask_input_ids,
            _mask_stacked_input_ids_v2,
            _pad_stacked_targets,
        )

        ls_len = token_res.ls_len or [len(in_dict["input_ids"])]

        add_eos = True
        if add_eos:
            input_ids = in_dict["input_ids"] + in_dict["labels"][-1:]
            len_extended_tokens = 1
        else:
            input_ids = in_dict["input_ids"]
            len_extended_tokens = 0

        # Handle ensemble datasets
        if len(gtokenizer.config.get("ensemble_datasets", [])) >= 2:
            assert (
                gtokenizer.sequence_packer is None
            ), "NOT implemented for packed sequences"
            reserved_semantics_token = gtokenizer.config["semantics"]["common"].get(
                "reserved_token", []
            )[graph.idx_of_ds]
            token_id = gtokenizer.vocab_map[reserved_semantics_token]
            ls_extend_tokens = [token_id]
            inputs_instance = input_ids[0]
            if isinstance(inputs_instance, list):
                ls_extend_tokens = [
                    [token_id] * len(inputs_instance) for token_id in ls_extend_tokens
                ]
            input_ids.extend(ls_extend_tokens)
            len_extended_tokens += len(ls_extend_tokens)

        # Only update ls_len[-1] if NOT using packed sequences
        if gtokenizer.sequence_packer is None:
            ls_len[-1] = ls_len[-1] + len_extended_tokens

        # Set up parameters for SMTP: scheduled masked token prediction
        mask_token_id = gtokenizer.get_mask_token_id()
        pad_token_id = gtokenizer.pad_token_id
        assert mask_token_id != pad_token_id

        conf = gtokenizer.config.get("pretrain_mlm", {})
        assert conf.get("name", "fixed") in {"polynomial", "cosine", "fixed"}
        mask_token_percent = conf.get("params", {}).get("mtp", 0.5)
        all_vocab_ids = gtokenizer.get_all_vocab_ids()

        # Mask input_ids and generate corresponding labels for training
        new_input_ids, new_labels_mask, wgts = [], [], []
        idx_left = 0

        for idx_right in ls_len:
            _input_ids = input_ids[idx_left:idx_right]
            idx_left = idx_right
            curr_mask_ratio, wgt = _get_mask_ratio(conf, gtokenizer)
            wgts.append(wgt)

            if isinstance(input_ids[0], list):
                if add_eos:
                    last_token_id = _input_ids[-1][0]
                    assert last_token_id == gtokenizer.get_eos_token_id()
                stack_method = getattr(gtokenizer, "stack_method", "short")
                _input_ids, _labels_mask = _mask_stacked_input_ids_v2(
                    _input_ids,
                    mask_token_id,
                    all_vocab_ids,
                    curr_mask_ratio,
                    mask_token_percent=mask_token_percent,
                    pad_token_id=pad_token_id,
                    has_eos=add_eos,
                    stack_method=stack_method,
                )
            else:
                assert isinstance(input_ids[0], int)
                last_token_id = _input_ids[-1]
                assert last_token_id == gtokenizer.get_eos_token_id()
                _input_ids, _labels_mask = _mask_input_ids(
                    _input_ids,
                    mask_token_id,
                    all_vocab_ids,
                    curr_mask_ratio,
                    mask_token_percent,
                    pad_token_id,
                )
            new_input_ids.extend(_input_ids)
            new_labels_mask.extend(_labels_mask)

        input_ids, labels_mask = new_input_ids, new_labels_mask

        # Add weights if configured
        if hasattr(gtokenizer, "train_cfg") and gtokenizer.train_cfg:
            if (
                hasattr(gtokenizer.train_cfg, "pretrain_mlm")
                and gtokenizer.train_cfg.pretrain_mlm.dlm_wgt
            ):
                in_dict["wgt"] = wgts

        # Handle "long" stack method
        if hasattr(gtokenizer, "stack_method") and gtokenizer.stack_method == "long":
            node_attr_dim = gtokenizer.config["semantics"]["node"]["dim"]
            labels_mask = [
                _pad_stacked_targets(
                    i, ls_labels, node_attr_dim=node_attr_dim, padding_val=-100
                )
                for i, ls_labels in enumerate(labels_mask)
            ]

        # Handle contrastive learning variant
        if gtokenizer.task_type == "pretrain-cl":
            input_ids, labels_mask, len_extended_tokens = self._add_gsum_tokens_for_cl(
                input_ids, labels_mask, gtokenizer, len_extended_tokens
            )

        in_dict["input_ids"] = input_ids
        in_dict["labels"] = labels_mask

        # Update attention mask and packed sequence info
        if gtokenizer.sequence_packer is None:
            in_dict["attention_mask"].extend([1] * len_extended_tokens)
            in_dict["split_lens"] = [len(in_dict["input_ids"])]
            in_dict["attn_modes"] = ["full"]
        else:
            lens = (np.array(ls_len) - np.array([0] + ls_len[:-1])).tolist()
            sequence_len = sum(lens)
            pad_len = gtokenizer.sequence_packer.mpe - sequence_len
            in_dict["split_lens"] = [int(l) for l in lens] + [pad_len]
            in_dict["sample_lens"] = [int(l) for l in lens] + [pad_len]
            in_dict["attn_modes"] = ["full"] * len(lens) + ["causal"]
            new_pos = []
            for l in lens:
                new_pos.extend(range(int(l)))
            in_dict["position_ids"] = new_pos
            in_dict["attention_mask"].extend([1] * len_extended_tokens)

        # Handle embeddings
        if "embed" in in_dict:
            dim = len(in_dict["embed"][0])
            extended_embed = np.zeros(
                (len_extended_tokens, dim), dtype=np.float32
            ).tolist()
            in_dict["embed"].extend(extended_embed)
            assert len(in_dict["embed"]) == len(in_dict["input_ids"])

        return in_dict

    def _add_gsum_tokens_for_cl(
        self, input_ids, labels_mask, gtokenizer, len_extended_tokens
    ):
        """Add gsum tokens for contrastive learning."""
        special_token_id = gtokenizer.get_gsum_token_id()
        ls_extend_tokens = [special_token_id]
        inputs_instance = input_ids[0]
        if isinstance(inputs_instance, list):
            ls_extend_tokens = [
                [token_id] * len(inputs_instance) for token_id in ls_extend_tokens
            ]
        input_ids.extend(ls_extend_tokens)

        label_pad_token_id = gtokenizer.label_pad_token_id
        ls_extend_labels = [label_pad_token_id]
        labels_mask_instance = labels_mask[0]
        if isinstance(labels_mask_instance, list):
            ls_extend_labels = [
                [token_id] * len(inputs_instance) for token_id in ls_extend_labels
            ]
        labels_mask.extend(ls_extend_labels)

        len_extended_tokens = len_extended_tokens + len(ls_extend_tokens)
        return input_ids, labels_mask, len_extended_tokens


class PretrainCoordStrategy(TaskPreparationStrategy):
    """Coordinate prediction pre-training preparation."""

    def prepare(self, in_dict, token_res, graph, gtokenizer):
        """Prepare inputs for coordinate prediction pre-training."""
        from ...masking import get_mask_of_raw_seq

        ls_raw_node_idx = token_res.ls_raw_node_idx

        input_ids = in_dict["input_ids"] + in_dict["labels"][-1:]  # add eos
        len_extended_tokens = 1

        assert len(gtokenizer.config.get("ensemble_datasets", [])) == 0
        assert gtokenizer.sequence_packer is None

        in_dict["input_ids"] = input_ids
        in_dict["attention_mask"].extend([1] * len_extended_tokens)

        # Handle embeddings
        if "embed" in in_dict:
            dim = len(in_dict["embed"][0])
            extended_embed = np.zeros(
                (len_extended_tokens, dim), dtype=np.float32
            ).tolist()
            in_dict["embed"].extend(extended_embed)
            assert len(in_dict["embed"]) == len(in_dict["input_ids"])

        # Attach node mask to inputs
        if ls_raw_node_idx is not None:
            input_ids = self._attach_node_mask_to_inputs(
                ls_raw_node_idx,
                len_extended_tokens,
                in_dict["input_ids"],
            )
            in_dict["input_ids"] = input_ids.tolist()

        in_dict["split_lens"] = [len(in_dict["input_ids"])]
        in_dict["attn_modes"] = ["full"]
        return in_dict

    def _attach_node_mask_to_inputs(
        self, ls_raw_node_idx, len_extended_tokens, input_ids
    ):
        """Attach node mask information to input IDs."""
        from ...masking import get_mask_of_raw_seq
        import numpy as np

        ls_raw_node_idx.extend([-1] * len_extended_tokens)
        node_idx = np.array(ls_raw_node_idx) + 1
        node_idx_clip = np.clip(node_idx, 0, 4)
        node_mask = get_mask_of_raw_seq(node_idx, mask_type="random")
        node_mask = node_mask * (node_idx > 0)
        edge_seq = list(zip([0] + node_idx.tolist()[:-1], node_idx.tolist()))
        edge_mask = get_mask_of_raw_seq(edge_seq, mask_type="random")
        edge_mask = edge_mask * (np.array(edge_seq) > 0).all(axis=-1)
        node_type = np.vstack([node_idx_clip, node_mask, node_idx, edge_mask]).T
        input_ids = np.hstack([np.array(input_ids), node_type])
        return input_ids
