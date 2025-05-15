import os
import random
import math
import numpy as np
from typing import Dict, List, Callable, Tuple, Optional, Union, Iterable, Set

import torch
from torch_geometric.data import Data
from torch.utils.data import IterableDataset, Dataset
from ..utils import (
    tokenizer_utils,
    nx_utils,
    instruct_tuning_utils,
    mol_utils,
    attn_mask_utils,
    graph2path,
    prepare_inputs_for_task,
    TASK_TYPES,
)
from .vocab_builder import load_vocab


DICT_pos_func = {
    "trans_rotate": mol_utils.trans_rotate_3d_random,
    "anchor_rotate": mol_utils.rotate_3d_v3,
}


class GSTTokenizer(object):
    def __init__(
        self,
        config: Dict,
        *,
        padding_side: str = "right",
        add_eos: bool = True,
        **kwargs,
    ):
        self.config = config
        self.mask_type = self.config["semantics"].get("attr_assignment", "random")
        assert padding_side in {"left", "right"}
        self.padding_side = padding_side
        self.add_eos = add_eos
        self.vocab_map = self.load_vocab()
        self.vocab_size = self.get_vocab_size()
        self.label_pad_token_id = -100
        self.pad_token_id = 0
        self.task_type = self.config["task_type"].lower()
        assert self.task_type in TASK_TYPES, f"{self.task_type} is not implemented!"
        self.eos_idx = None
        self.semantics2tokens_mapping = get_semantics_raw_node_edge2attr_mapping
        # below for pack target token sequence with randomly sampled token sequence
        self.mpe = None
        self.dataset = None
        self.sampler = None
        self.token_components = None
        self.random_ratio = 1
        self.label_to_be_padded = self.get_label_token_id_to_be_padded()
        # cyclic mpe
        self.cmpe = 100000
        self.cyclic_mpe = False
        self.node_idx_tokens = None
        self.node_idx_token_ids = None
        self.all_token_ids = None
        # kwargs
        self.kwargs = kwargs

    def load_vocab(self):
        fn = os.path.join(
            self.config["name_or_path"], self.config.get("vocab_file", "vocab")
        )
        return load_vocab(fn)

    def get_vocab_size(self):
        return max(self.vocab_map.values()) + 1

    def get_all_vocab_ids(self):
        if self.all_token_ids is None:
            self.all_token_ids = tuple(range(self.get_vocab_size()))
        return self.all_token_ids

    def build_vocab(self):
        pass

    def get_label_pad_token(self):
        return "<label_pad>"

    def get_bos_token(self):
        return self.config["structure"]["node"]["bos_token"]

    def get_eos_token(self):
        return self.config["structure"]["node"]["eos_token"]

    def get_new_node_token(self):
        return self.config["structure"]["node"].get(
            "new_node_token", self.get_label_pad_token()
        )

    def get_edge_bi_token(self):
        return self.config["structure"]["edge"]["bi_token"]

    def get_jump_token(self):
        return self.config["structure"]["edge"]["jump_token"]

    def get_gsum_token(self):
        return self.config["structure"]["graph"]["summary_token"]

    def get_mask_token(self):
        return self.config["structure"]["common"].get("mask_token", "<mask>")

    def get_icl_token(self):
        return self.config["structure"]["common"]["icl_token"]

    def get_sep_token(self):
        return self.config["structure"]["common"]["sep_token"]

    def get_common_structure(self):
        return self.config["structure"]["common"].get("reserved_token", [])

    def get_common_semantics(self):
        return self.config["semantics"]["common"].get("reserved_token", [])

    def get_bos_token_id(self):
        return self.vocab_map[self.get_bos_token()]

    def get_eos_token_id(self):
        return self.vocab_map[self.get_eos_token()]

    def get_new_node_token_id(self):
        return self.vocab_map[self.get_new_node_token()]

    def get_jump_token_id(self):
        return self.vocab_map[self.get_jump_token()]

    def get_gsum_token_id(self):
        return self.vocab_map.get(self.get_gsum_token(), None)

    def get_mask_token_id(self):
        return self.vocab_map[self.get_mask_token()]

    def get_node_idx_tokens(self):
        if self.node_idx_tokens is None:
            self.node_idx_tokens = {
                str(x) for x in range(self.config["structure"]["node"]["scope_base"])
            }
        return self.node_idx_tokens

    def get_node_idx_token_ids(self):
        if self.node_idx_token_ids is None:
            self.node_idx_token_ids = {
                self.vocab_map[str(x)]
                for x in range(self.config["structure"]["node"]["scope_base"])
            }
        return self.node_idx_token_ids

    def get_token_components(self, ls_tokens):
        if self.token_components is None:
            one_token = ls_tokens[0]
            if isinstance(one_token, List):
                self.token_components = len(one_token)
            else:
                self.token_components = 0
        return self.token_components

    def get_label_token_id_to_be_padded(self):
        if self.task_type != "pretrain":
            label_token_ids = set([])
        else:
            label_tokens_to_be_padded = set(self.config.get("label_tokens_to_pad", []))
            label_token_ids = [
                self.vocab_map[token] for token in label_tokens_to_be_padded
            ]
            label_token_ids = set(label_token_ids)
        print(
            f"label token id to be converted to {self.label_pad_token_id} is {label_token_ids}"
        )
        return label_token_ids

    def _map_tokens_to_ids(self, tokens: Union[str, Iterable[str]]):
        if tokens is None:
            token_ids = None
        elif isinstance(tokens, str):
            token_ids = self.vocab_map[tokens]
        elif isinstance(tokens, Iterable):
            token_ids = tuple([self.vocab_map[token] for token in tokens])
        else:
            raise NotImplementedError(
                f"Not implement for type {type(tokens)} of tokens {tokens}!"
            )
        return token_ids

    def encode(self, seq):
        # input: tokenized sequence
        # output: id in vocabulary
        pass

    def set_eos_idx(self, input_ids: List[Union[int, List[int]]]):
        if self.eos_idx is None:
            # every worker of Loader will calculate eos_idx independently!
            eos_token_id = self.get_eos_token_id()
            assert isinstance(input_ids, list)
            try:
                if self.mpe is None:
                    # NO packing of token sequence
                    if isinstance(input_ids[0], List):
                        input_ids = [x[0] for x in input_ids]
                    if self.task_type == "nodev2":
                        idx = input_ids.index(eos_token_id)
                        self.eos_idx = idx - len(input_ids)
                    else:
                        self.eos_idx = int(1e8)
                else:
                    # PACKING token sequences with <eos_token> as sep
                    self.eos_idx = int(1e8)
            except ValueError as e:
                # print(f"ValueError: {e}:: {input_ids}")
                self.eos_idx = int(1e8)
            finally:
                print(
                    f"[Warning] Set eos_idx to {self.eos_idx} for task {self.task_type}!"
                )
            # print(f"SET eos_idx to be {self.eos_idx}")
            # This will be executed every epoch at every worker of the loader

    def pad(
        self,
        features: List[Dict],
        padding: bool = True,
        max_length: int = 128,
        pad_to_multiple_of: int = 8,
        return_tensors: str = "pt",
        mask_boundary: bool = False,
    ):
        # features:: list of input dicts
        # params setting is compatible with HF transformers
        self.set_eos_idx(features[0]["input_ids"])
        assert return_tensors in {"pt", "np"}
        func = {"pt": torch.tensor, "np": np.array}[return_tensors]
        ls_seq_len = [len(feat["input_ids"]) for feat in features]
        pad_to = _get_batch_seq_len(ls_seq_len, pad_to_multiple_of, max_length)
        features = [self._pad_each_datapoint(feat, pad_to) for feat in features]

        batch_outputs = {}
        for feat in features:
            for key, value in feat.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        batch_outputs = {
            key: func(val) if not isinstance(val[0], str) else np.array(val)
            for key, val in batch_outputs.items()
        }
        if "attention_mask_bi" in batch_outputs:
            attention_mask = batch_outputs["attention_mask"]
            attention_mask_bi = batch_outputs["attention_mask_bi"]
            assert (
                attention_mask | attention_mask_bi
            ).sum().item() == attention_mask.sum().item()
            if mask_boundary:
                # The calculation is time-consuming, so it is moved here
                mask_idx = attn_mask_utils.get_masked_boundary_idx(
                    attention_mask, attention_mask_bi, pad_to
                )
                batch_outputs["boundary_mask_idx"] = mask_idx
        return batch_outputs

    def _pad_each_datapoint(self, feature, pad_to):
        if pad_to > len(feature["input_ids"]):
            padding_len = pad_to - len(feature["input_ids"])

            if isinstance(feature["input_ids"][0], Iterable):
                input_pad_val = [self.pad_token_id] * len(feature["input_ids"][0])
            else:
                assert isinstance(feature["input_ids"][0], int)
                input_pad_val = self.pad_token_id
            if isinstance(feature["labels"][0], Iterable):
                label_pad_val = [self.label_pad_token_id] * len(feature["labels"][0])
            else:
                assert isinstance(feature["labels"][0], int)
                label_pad_val = self.label_pad_token_id
            padded_input_ids = [input_pad_val] * padding_len
            padded_labels = [label_pad_val] * padding_len

            padded_nodev2_labels = [self.label_pad_token_id] * padding_len
            padded_position_ids = [0] * padding_len
            padded_attention_mask = [0] * padding_len

            feature["input_ids"] = _merge_two_ls(
                feature["input_ids"], padded_input_ids, self.padding_side
            )
            feature["position_ids"] = _merge_two_ls(
                feature["position_ids"], padded_position_ids, self.padding_side
            )
            feature["labels"] = _merge_two_ls(
                feature["labels"], padded_labels, self.padding_side
            )
            assert isinstance(
                feature["attention_mask"], List
            ), f"attention_mask type: {type(feature['attention_mask'])}"
            feature["attention_mask"] = _merge_two_ls(
                feature["attention_mask"], padded_attention_mask, self.padding_side
            )
            if "nodev2_labels" in feature:
                feature["nodev2_labels"] = _merge_two_ls(
                    feature["nodev2_labels"], padded_nodev2_labels, self.padding_side
                )
            if "raw_node_idx" in feature:
                feature["raw_node_idx"] = _merge_two_ls(
                    feature["raw_node_idx"], padded_nodev2_labels, self.padding_side
                )
            if "attention_mask_bi" in feature:
                feature["attention_mask_bi"] = _merge_two_ls(
                    feature["attention_mask_bi"],
                    padded_attention_mask,
                    self.padding_side,
                )
            set_vectors = {"embed", "noise"}
            for name in set_vectors:
                if name in feature:
                    dim = len(feature[name][0])
                    padded_vecs = np.zeros(
                        (padding_len, dim), dtype=np.float32
                    ).tolist()
                    feature[name] = _merge_two_ls(
                        feature[name], padded_vecs, self.padding_side
                    )
        else:
            keys_set = {
                "input_ids",
                "position_ids",
                "labels",
                "nodev2_labels",
                "raw_node_idx",
                "attention_mask",
                "attention_mask_bi",
                "embed",
                "noise",
            }
            # eos_idx<0 or eos_idx=1e8
            # the design of negative `eos_idx` is to keep the task-specific tails
            # e.g., tail `<eos> <tgt-node>` for node task -> eos_idx=-2
            # tail `<eos> <src-node> <dst-node>` for edge task -> eos_idx=-3
            # tail `<eos> <gsum>` for graph task -> eos_idex=-2
            mid_idx = pad_to + self.eos_idx if self.eos_idx < 0 else pad_to
            for key, val in feature.items():
                if key in keys_set:
                    if isinstance(val, np.ndarray):
                        # For 2D block-wise attention mask in pre-train with packed sequence
                        assert key == "attention_mask", f"NOT for {key}"
                        feature[key] = val[:pad_to, :pad_to].tolist()
                    else:
                        feature[key] = val[:mid_idx] + val[self.eos_idx :]
            if ("cls_idx" in feature) and (self.eos_idx < 0):
                feature["cls_idx"] = [pad_to + self.eos_idx + 1]
        return feature

    def pack_token_seq(
        self, token_res: tokenizer_utils.TokenizationOutput, previous_idx: int
    ):
        ls_tokens = token_res.ls_tokens
        ls_labels = token_res.ls_labels
        ls_embed = token_res.ls_embed
        token_compontens = self.get_token_components(ls_tokens)
        token_len = len(ls_tokens) + 1
        ls_len = [token_len]
        if isinstance(self.dataset, IterableDataset):
            print(f"Convert Iterable dataset to iterable: `dataset -> iter(dataset)`")
            self.dataset = iter(self.dataset)
        while token_len < self.mpe:
            if isinstance(self.dataset, Dataset):
                if random.uniform(0, 1.0) <= self.random_ratio:
                    # randomly sample a graph
                    idx = (
                        self.dataset.get_random_sample_idx()
                        if hasattr(self.dataset, "get_random_sample_idx")
                        else random.choice(self.sampler)
                    )
                else:  # repeat the previous graph
                    idx = previous_idx
                # sep_token = (
                #     self.get_eos_token()
                #     if idx != previous_idx
                #     else self.get_gsum_token()
                # )  # causing problem when pretrain-mlm
                sep_token = self.get_eos_token()
                _, new_graph = self.dataset[idx]
            else:
                idx = 0
                sep_token = self.get_eos_token()
                _, new_graph = next(self.dataset)
            token_res = self.tokenize(new_graph)
            new_ls_tokens = token_res.ls_tokens
            new_ls_labels = token_res.ls_labels
            new_ls_embed = token_res.ls_embed
            if token_compontens == 0:
                seps = [sep_token]
                label_seps = [self.get_label_pad_token()]
            else:
                seps = [[sep_token] * token_compontens]
                label_seps = [[self.get_label_pad_token()] * token_compontens]
            embed_seps = []
            if ls_embed:
                dim = len(ls_embed[0])
                embed_seps = np.zeros((1, dim), dtype=np.float32).tolist()
            ls_tokens = ls_tokens + seps + new_ls_tokens
            ls_labels = ls_labels + label_seps + new_ls_labels
            if ls_embed:
                ls_embed = ls_embed + embed_seps + new_ls_embed

            previous_idx = idx
            token_len = len(ls_tokens) + 1
            ls_len.append(token_len)
        return ls_tokens, ls_labels, ls_embed, ls_len

    def _tailor_node_struct_repr(self, node_structure_mapping):
        # for forward compatibility
        return node_structure_mapping

    def _tailor_edge_struct_repr(self, edge_structure_mapping):
        # for forward compatibility
        return edge_structure_mapping

    def tokenize(self, graph: Data):
        return self.raw_tokenize(graph)

    def raw_tokenize(self, graph: Data):
        # input: raw small/medium graph OR subgraph sampled from big graphs
        # output: sequence of tokens from vocab
        # 1~2. transform graph to Eulerian sequence
        assert (
            graph.num_nodes <= self.config["structure"]["node"]["node_scope"]
        ), f"num_nodes: {graph.num_nodes} > node_scope: {self.config['structure']['node']['node_scope']}"
        path = graph2path(graph)
        # 3. obtain node/edge structure and semantics mapping
        node_structure_mapping = nx_utils.get_structure_raw_node2idx_mapping(
            path,
            self.config["structure"]["node"]["scope_base"],
            self.config["structure"]["node"]["node_scope"],
            self.config["structure"]["node"].get("cyclic", False),
        )
        node_structure_mapping = self._tailor_node_struct_repr(node_structure_mapping)
        edge_structure_mapping = nx_utils.get_structure_raw_edge2type_mapping(
            path, graph
        )
        edge_structure_mapping = self._tailor_edge_struct_repr(edge_structure_mapping)
        (
            node_semantics_mapping,
            edge_semantics_mapping,
            graph_semantics_mapping,
        ) = self.semantics2tokens_mapping(path, graph, self.config)
        # 3.1 obtain target node or target edge tokens FOR node/edge-lvl tasks
        tgt_node_token = None
        tgt_edge_src_token = None
        tgt_edge_dst_token = None
        if hasattr(graph, "root_n_id"):
            # use re-indexed node-id to repr the node, e.g., 0/1/2/3/...
            if isinstance(graph.root_n_id, int):
                tgt_node_token = node_structure_mapping[graph.root_n_id]
            elif (
                isinstance(graph.root_n_id, torch.Tensor) and len(graph.root_n_id) == 2
            ):
                src, dst = graph.root_n_id.tolist()
                tgt_edge_src_token = node_structure_mapping[src]
                tgt_edge_dst_token = node_structure_mapping[dst]
            else:
                raise ValueError(
                    f"graph.root_n_id {graph.root_n_id} is not supported, Please check!"
                )
        # 4. decorate node/edge/graph with above mapping
        raw_seq = nx_utils.get_raw_seq_from_path(path)
        mask = tokenizer_utils.get_mask_of_raw_seq(raw_seq, self.mask_type)
        (
            ls_tokens,
            ls_node_regression_labels,
            ls_edge_regression_labels,
        ) = nx_utils.decorate_node_edge_graph_with_mask(
            self,
            raw_seq,
            mask,
            node_structure_mapping,
            edge_structure_mapping,
            node_semantics_mapping,
            edge_semantics_mapping,
            graph_semantics_mapping,
            attr_shuffle=self.config["semantics"].get("attr_shuffle", False),
        )
        # 4.5 DEPRECATED@2024-10-22:: special decoration for molecule data with 3d coordinates
        # 5. remove bidirectional edge-type token, because it is treated as default edge-type,
        # keeping it will produce lots of redundant tokens
        dict_edge = self.config["structure"]["edge"]
        if dict_edge.get("remove_edge_type_token", False):
            edge_types = {self._tailor_edge_struct_repr(dict_edge["bi_token"])}
            ls_tokens = [token for token in ls_tokens if token not in edge_types]
        # 5.1 obtain label tokens from input tokens
        ls_labels = nx_utils.get_labels_from_input_tokens(ls_tokens, self)
        # 6. add nx/instructions/eos tokens and etc.
        # 6.1 enable nx func to enhance structure understanding
        ls_struct_tokens, ls_struct_labels = nx_utils.understand_structure(
            graph,
            tokenization_config=self.config,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            gtokenizer=self,
        )
        ls_tokens.extend(ls_struct_tokens)
        ls_labels.extend(ls_struct_labels)
        # 6.2 enable instruction tuning to enhance semantics understanding
        (
            ls_instruct_tokens,
            ls_instruct_labels,
        ) = instruct_tuning_utils.follow_instructions(
            graph,
            tokenization_config=self.config,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            gtokenizer=self,
        )
        ls_tokens.extend(ls_instruct_tokens)
        ls_labels.extend(ls_instruct_labels)
        # 6.3 add special tokens, e.g., eos
        if self.add_eos and ("pretrain" not in self.task_type):
            ls_tokens = self.add_eos_token(ls_tokens)
            ls_labels = self.add_eos_token(ls_labels)
        return tokenizer_utils.TokenizationOutput(
            ls_tokens=ls_tokens,
            ls_labels=ls_labels,
            tgt_node_token=tgt_node_token,
            tgt_edge_src_token=tgt_edge_src_token,
            tgt_edge_dst_token=tgt_edge_dst_token,
            tgt_pos=None,  # TODO: implement ls_embed for GSTTokenizer
        )

    def convert_tokens_to_ids(self, seq_tokens, seq_labels):
        # 7. map tokens to token-id
        seq_tokens_id = [self.vocab_map[token] for token in seq_tokens]
        seq_labels_id = [self.vocab_map[token] for token in seq_labels]
        # 8. add labels, attention mask, position_ids and etc
        in_dict = get_input_dict_from_seq_tokens_id(
            seq_tokens_id,
            seq_labels_id,
            self.label_to_be_padded,
            self.label_pad_token_id,
            self.cmpe,
            self.cyclic_mpe,
            self.get_node_idx_token_ids().union(
                {
                    self.get_eos_token_id(),
                    self.get_jump_token_id(),
                    self.get_gsum_token_id(),
                }
            ),
        )
        return in_dict

    def prepare_inputs_for_task(
        self,
        in_dict: Dict,
        graph: Data,
        token_res: tokenizer_utils.TokenizationOutput,
        is_training: Optional[bool] = None,
    ):
        tgt_node_token_id = self._map_tokens_to_ids(token_res.tgt_node_token)
        tgt_edge_src_token_id = self._map_tokens_to_ids(token_res.tgt_edge_src_token)
        tgt_edge_dst_token_id = self._map_tokens_to_ids(token_res.tgt_edge_dst_token)
        in_dict = prepare_inputs_for_task(
            self.task_type,
            in_dict,
            graph=graph,
            eos_token_id=self.get_eos_token_id(),
            tgt_node_token_id=tgt_node_token_id,
            tgt_edge_src_token_id=tgt_edge_src_token_id,
            tgt_edge_dst_token_id=tgt_edge_dst_token_id,
            tgt_pos=token_res.tgt_pos,
            gsum_token_id=self.get_gsum_token_id(),
            gtokenizer=self,
            ls_len=token_res.ls_len,
            ls_raw_node_idx=token_res.ls_raw_node_idx,
        )
        return in_dict

    def __call__(self, graph: Data, is_training: Optional[bool] = None):
        # 1~6. self.tokenize
        token_res = self.tokenize(graph)
        ls_tokens, ls_labels, ls_embed, ls_len = (
            self.pack_token_seq(token_res, graph.idx)
            if self.mpe is not None
            else (
                token_res.ls_tokens,
                token_res.ls_labels,
                token_res.ls_embed,
                [len(token_res.ls_tokens)],
            )
        )
        # 7~8. self.convert_tokens_to_ids
        in_dict = self.convert_tokens_to_ids(ls_tokens, ls_labels)
        if ls_embed:
            in_dict["embed"] = ls_embed
        # 9. prepare for tasks
        token_res.ls_tokens = ls_tokens
        token_res.ls_labels = ls_labels
        token_res.ls_embed = ls_embed
        token_res.ls_len = ls_len
        in_dict = self.prepare_inputs_for_task(
            in_dict,
            graph,
            token_res=token_res,
        )
        return in_dict

    def add_eos_token(self, seq_tokens):
        eos_token = self.get_eos_token()
        seq_tokens.append(eos_token)
        return seq_tokens

    def save_pretrained(self):
        pass


def _merge_two_ls(ls_main, ls_side, side="left"):
    return ls_side + ls_main if side == "left" else ls_main + ls_side


def _get_batch_seq_len(ls_seq_len, pad_to_multiple_of, max_position_embeddings):
    if pad_to_multiple_of is None:
        batch_seq_len = max_position_embeddings
    else:
        max_seq_len = max(ls_seq_len)
        batch_seq_len = pad_to_multiple_of * int(
            math.ceil(max_seq_len / pad_to_multiple_of)
        )
        batch_seq_len = min(batch_seq_len, max_position_embeddings)
    return batch_seq_len


def get_input_dict_from_seq_tokens_id(
    seq_tokens_id: List[Union[int, List[int]]],
    seq_labels_id: List[Union[int, List[int]]],
    label_to_be_pad: Set[int],
    label_pad_token_id: int,
    mpe: int = 1024,
    cyclic: bool = False,
    node_idx_token_ids: Set = set(),
):
    seq_tokens_id = list(seq_tokens_id)
    if seq_labels_id is None:
        input_ids = seq_tokens_id[:-1]
        labels = seq_tokens_id[1:]
    else:
        seq_labels_id = list(seq_labels_id)
        input_ids = seq_tokens_id
        labels = seq_labels_id
    assert len(input_ids) == len(
        labels
    ), f"input_ids: {len(input_ids)}, labels: {len(labels)}"
    if len(label_to_be_pad) > 0:
        labels = [
            token_id if token_id not in label_to_be_pad else label_pad_token_id
            for token_id in labels
        ]
    # `random.randint` Return random integer in range [a, b], including both end points.
    # v1: cyclic version
    start_idx = random.randint(0, mpe - 1) if cyclic else 0
    position_ids = list(
        [ele % mpe for ele in range(start_idx, start_idx + len(input_ids))]
    )
    # v2: non-cyclic version
    # start_idx = random.randint(0, max(0, mpe - len(input_ids) - 5)) if cyclic else 0
    # position_ids = list(range(start_idx, start_idx + len(input_ids)))
    # v3: pe relying on node positions
    if cyclic:  # TODO: use another params to turn on/off this functionality
        ls_tf = [1 if x in node_idx_token_ids else 0 for x in input_ids]
        ls_tf = [1] * len(input_ids) if not isinstance(input_ids[0], int) else ls_tf
        position_ids = (np.cumsum(ls_tf) - 1).tolist()

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def _tokenize_discrete_attr(
    raw_attr: List[str],
    world_identifier: str,
    node_edge_identifier: str,
    ignored_val: str = None,
    shuffle: bool = False,
    remove_val: bool = False,
    share_vocab: bool = False,
):
    # input:: raw_attr: e.g., [4932, 29376]
    # output:: e.g., ['ogbn-proteins#node#0#4932', 'ogbn-proteins#node#1#29376']
    #            OR  ['ogbn-proteins#node#0', 'ogbn-proteins#node#1']
    if remove_val:
        tokens = [
            f"{world_identifier}#{node_edge_identifier}#-1"
            if share_vocab
            else f"{world_identifier}#{node_edge_identifier}#{col_idx}"
            for col_idx, _ in enumerate(raw_attr)
        ]
    else:
        tokens = [
            f"{world_identifier}#{node_edge_identifier}#-1#{col_val}"
            if share_vocab
            else f"{world_identifier}#{node_edge_identifier}#{col_idx}#{col_val}"
            for col_idx, col_val in enumerate(raw_attr)
            if col_val != str(ignored_val)
        ]
        if shuffle:
            random.shuffle(tokens)
    return tokens


def _remove_lead_zero(ls_col_val):
    # remove leading 0 to reduce token length if it is decimals < 1
    return (
        ls_col_val[1:]
        if (len(ls_col_val) > 2) and (ls_col_val[0] == "0") and (ls_col_val[1] == ".")
        else ls_col_val
    )


def _tokenize_continuous_attr(
    raw_attr: List[str],
    world_identifier: str,
    node_edge_identifier: str,
    ignored_val: str = None,
    shuffle: bool = False,
):
    # input:: raw_attr: e.g., ['500', '0', '380']
    # output:: e.g., ['ogbn-proteins#node#0#1', '<5>', '<0>', '<0>', 'ogbn-proteins#node#2#1', '<3>', '<8>', '<0>']
    def _process_each_col(col_idx, col_val):
        ls_col_val = list(col_val)
        ls_col_val = _remove_lead_zero(ls_col_val)
        ls_col_val = [f"<{x}>" for x in ls_col_val]
        identifier = (
            f"{world_identifier}#{node_edge_identifier}#{col_idx}#1"
            if node_edge_identifier != "graph"
            else "<gsum>"
        )
        return [identifier] + ls_col_val

    tokens = [
        _process_each_col(col_idx, col_val)
        for col_idx, col_val in enumerate(raw_attr)
        if col_val != str(ignored_val)
    ]
    if shuffle:
        random.shuffle(tokens)
    return tokens


def _add_regression_token(dict_map, reg_token):
    for val in dict_map.values():
        val.append(reg_token)


def _get_node2attr_mapping(path, data: Data, attr_name: str):
    if path:
        tmp_map = {}
        for src, tgt in path:
            if tmp_map.get(src, None) is None:
                tmp_map[src] = data[attr_name][src].numpy()

        # for semi-euler path OR shortened euler path, which does not go back to origin
        if tmp_map.get(tgt, None) is None:
            tmp_map[tgt] = data[attr_name][tgt].numpy()
    else:  # in case `path=[]` when graph has ONLY 1 node
        node = 0
        tmp_map = {node: data[attr_name][node].numpy()}
    return tmp_map


def _get_edge2attr_mapping(path, data: Data, attr_name: str, verbose: bool = False):
    tmp_map = {}
    for src, tgt in path:
        idx = nx_utils.get_edge_index(data.edge_index, src, tgt)
        if idx.shape[0] == 0:
            idx_backward = nx_utils.get_edge_index(data.edge_index, tgt, src)
            if idx_backward.shape[0] == 0:
                idx = None
                print(
                    f"Edge ({src}, {tgt}) or ({tgt}, {src}) does not have attr {attr_name}"
                ) if verbose else None
            else:
                idx = idx_backward
        if idx is not None:
            idx = idx.item()
            attr_val = data[attr_name][idx]
            tmp_map[(src, tgt)] = attr_val.numpy()
    return tmp_map


def _get_graph2attr_mapping(path, data: Data, attr_name: str):
    return {0: data[attr_name][0].numpy()}


def get_semantics_attr_mapping(
    path, data: Data, config: Dict, node_or_edge: str, func_attr_mapping: Callable
):
    # input: path
    # output: a mapping of each node/edge to its attr, and each node to its global-idx if exists
    assert node_or_edge in {"node", "edge", "graph"}
    dict_map = {"discrete": {}, "continuous": {}, "embed": {}}
    attr_shuffle = config["semantics"].get("attr_shuffle", False)

    discrete_attr = config["semantics"][node_or_edge]["discrete"]
    world_identifier = config["attr_world_identifier"]
    if discrete_attr is not None:
        share_vocab = config["semantics"][node_or_edge].get("share_vocab", False)
        ignored_val = config["semantics"][node_or_edge]["ignored_val"]
        tmp_map = func_attr_mapping(path, data, discrete_attr)
        dict_map["discrete"] = (
            {
                k: _tokenize_discrete_attr(
                    v.astype(str),
                    world_identifier,
                    node_or_edge,
                    ignored_val,
                    attr_shuffle,
                    share_vocab=share_vocab,
                )
                for k, v in tmp_map.items()
            }
            if tmp_map
            else {(-1, -1): None}
        )

    continuous_attr = config["semantics"][node_or_edge]["continuous"]
    if continuous_attr is not None:
        assert (
            discrete_attr is None
        ), "Supporting both discrete and continuous attr is NOT implemented yet!"
        ignored_val = config["semantics"][node_or_edge]["ignored_val"]
        tmp_map = func_attr_mapping(path, data, continuous_attr)
        dict_map["discrete"] = {
            k: _tokenize_continuous_attr(
                v.astype(str), world_identifier, node_or_edge, ignored_val, attr_shuffle
            )
            for k, v in tmp_map.items()
        }

    embed_attr = config["semantics"][node_or_edge].get("embed", None)
    if embed_attr is not None:
        tmp_map = func_attr_mapping(path, data, embed_attr)
        dict_map["embed"] = (
            {k: v.tolist() for k, v in tmp_map.items()} if tmp_map else {(-1, -1): None}
        )
    return dict_map


def get_semantics_raw_node_edge2attr_mapping(path, data: Data, config: Dict):
    dict_map_node = get_semantics_attr_mapping(
        path, data, config, "node", _get_node2attr_mapping
    )
    dict_map_edge = get_semantics_attr_mapping(
        path, data, config, "edge", _get_edge2attr_mapping
    )
    dict_map_graph = get_semantics_attr_mapping(
        path, data, config, "graph", _get_graph2attr_mapping
    )
    return dict_map_node, dict_map_edge, dict_map_graph


def mask_semantics_attr(data: Data, config: Dict, node_or_edge: str):
    # input: path
    # output: a mapping of each node/edge to its attr, and each node to its global-idx if exists
    assert node_or_edge in {"node", "edge", "graph"}

    discrete_attr = config["semantics"][node_or_edge]["discrete"]
    if discrete_attr is not None:
        data[discrete_attr] = data[discrete_attr] * 0

    continuous_attr = config["semantics"][node_or_edge]["continuous"]
    if continuous_attr is not None:
        assert (
            discrete_attr is None
        ), "Supporting both discrete and continuous attr is NOT implemented yet!"
        data[continuous_attr] = data[continuous_attr] * 0
    return data


def mask_semantics_raw_node_edge_attr(data: Data, config: Dict):
    data = data.clone()
    data = mask_semantics_attr(data, config, "node")
    data = mask_semantics_attr(data, config, "edge")
    data = mask_semantics_attr(data, config, "graph")
    return data


class StackedGSTTokenizer(GSTTokenizer):
    def __init__(
        self,
        config: Dict,
        *,
        padding_side: str = "right",
        add_eos: bool = True,
        stack_method: str = "short",
        rotation: str = "anchor_rotate",
        **kwargs,
    ):
        super().__init__(config, padding_side=padding_side, add_eos=add_eos, **kwargs)
        assert stack_method in {"short", "long"}
        self.stack_method = stack_method
        self.default_node_attr = None
        self.default_edge_attr = None
        self.default_node_embed = None
        self.default_edge_embed = None
        self.default_edge_attr_id = None
        self.config["semantics"]["node"]["ignored_val"] = None
        self.config["semantics"]["edge"]["ignored_val"] = None
        assert rotation in DICT_pos_func.keys(), f"your rotation: {rotation}"
        self.rotation = rotation
        print(
            f"[StackedGSTTokenizer] stack_method: {stack_method}, rotation: {rotation}"
        )

    def get_default_node_attr(self, graph: Optional[Data] = None):
        if self.default_node_attr is None:
            self.default_node_attr = get_default_semantics_attr_mapping(
                graph, self.config, "node"
            )
        return self.default_node_attr

    def get_default_edge_attr(self, graph: Optional[Data] = None):
        if self.default_edge_attr is None:
            self.default_edge_attr = get_default_semantics_attr_mapping(
                graph, self.config, "edge"
            )
        return self.default_edge_attr

    def get_default_edge_attr_id(self, graph: Optional[Data] = None):
        if self.default_edge_attr_id is None:
            default_edge_attr = self.get_default_edge_attr(graph)
            self.default_edge_attr_id = self._map_tokens_to_ids(default_edge_attr)
        return self.default_edge_attr_id

    def get_default_node_embed(self, graph: Optional[Data] = None):
        if self.default_node_embed is None:
            self.default_node_embed = get_default_semantics_embed_mapping(
                graph, self.config, "node"
            )
        return self.default_node_embed

    def get_default_edge_embed(self, graph: Optional[Data] = None):
        if self.default_edge_embed is None:
            self.default_edge_embed = get_default_semantics_embed_mapping(
                graph, self.config, "edge"
            )
        return self.default_edge_embed

    def add_eos_token(self, ls_tokens):
        eos_token = self.config["structure"]["node"]["eos_token"]
        # ls = [eos_token] + self.get_default_node_attr() + self.get_default_edge_attr()
        ls = [eos_token] * len(ls_tokens[0])
        ls_tokens.append(ls)
        return ls_tokens

    def get_tokens_from_single_edge_attr(self, edge_attr: torch.Tensor):
        assert len(edge_attr.shape) == 1
        tokens = []
        v = edge_attr.numpy()
        node_or_edge = "edge"
        discrete_attr = self.config["semantics"][node_or_edge]["discrete"]
        world_identifier = self.config["attr_world_identifier"]
        if discrete_attr is not None:
            share_vocab = self.config["semantics"][node_or_edge].get(
                "share_vocab", False
            )
            ignored_val = self.config["semantics"][node_or_edge]["ignored_val"]
            tokens = _tokenize_discrete_attr(
                v.astype(str),
                world_identifier,
                node_or_edge,
                ignored_val=ignored_val,
                shuffle=False,
                share_vocab=share_vocab,
            )
        return tokens

    def tokenize(self, graph: Data):
        # input: raw small/medium graph OR subgraph sampled from big graphs
        # output: sequence of tokens from vocab
        if hasattr(graph, "pos") and graph.pos is not None:
            graph.pos = DICT_pos_func[self.rotation](graph.pos)
        if hasattr(graph, "rdkit_pos"):
            graph.rdkit_pos = DICT_pos_func[self.rotation](graph.rdkit_pos)
            graph.pos = torch.hstack([graph.pos, graph.rdkit_pos])
        # 1 & 2. get eulerian path
        path = graph2path(graph, prioritize=self.task_type != "pretrain")
        # 3. obtain node/edge structure and semantics mapping
        node_structure_mapping = nx_utils.get_structure_raw_node2idx_mapping(
            path,
            self.config["structure"]["node"]["scope_base"],
            self.config["structure"]["node"]["node_scope"],
            self.config["structure"]["node"].get("cyclic", False),
        )
        edge_structure_mapping = nx_utils.get_structure_raw_edge2type_mapping(
            path, graph
        )
        (
            node_semantics_mapping,
            edge_semantics_mapping,
            graph_semantics_mapping,
        ) = get_semantics_raw_node_edge2attr_mapping(path, graph, self.config)
        # below to be compatible with `instruct_tuning_utils._get_all_node_feats`
        node_structure_mapping[-1] = (self.get_new_node_token(),)
        edge_structure_mapping[(-1, -1)] = self.get_edge_bi_token()
        if node_semantics_mapping["discrete"]:
            node_semantics_mapping["discrete"][-1] = self.get_default_node_attr(graph)
        if edge_semantics_mapping["discrete"]:
            edge_semantics_mapping["discrete"][(-1, -1)] = self.get_default_edge_attr(
                graph
            )
        if node_semantics_mapping["embed"]:
            node_semantics_mapping["embed"][-1] = self.get_default_node_embed(graph)
        if edge_semantics_mapping["embed"]:
            edge_semantics_mapping["embed"][(-1, -1)] = self.get_default_edge_embed(
                graph
            )
        # 3.1 obtain target node or target edge tokens FOR node/edge-lvl tasks
        tgt_node_token = None
        tgt_edge_src_token = None
        tgt_edge_dst_token = None
        tgt_edge_attr_token = None
        if hasattr(graph, "root_n_id"):
            # use re-indexed node-id to repr the node, e.g., 0/1/2/3/...
            if isinstance(graph.root_n_id, int):
                tgt_node_token = node_structure_mapping[graph.root_n_id]
            elif (
                isinstance(graph.root_n_id, torch.Tensor) and len(graph.root_n_id) == 2
            ):
                src, dst = graph.root_n_id.tolist()
                tgt_edge_src_token = node_structure_mapping[src]
                tgt_edge_dst_token = node_structure_mapping[dst]
                if hasattr(graph, "tgt_edge_attr"):
                    tgt_edge_attr_token = self.get_tokens_from_single_edge_attr(
                        graph.tgt_edge_attr
                    )
            else:
                raise ValueError(
                    f"graph.root_n_id {graph.root_n_id} is not supported, Please check!"
                )
        if self.task_type == "nodev2":
            assert tgt_node_token is None
            tgt_node_token = nx_utils._flatten_list(
                [node_structure_mapping[ele] for ele in range(graph.num_nodes)]
            )
        # 4. remove bidirectional edge-type token, because it is treated as default edge-type,
        # keeping it will produce redundant tokens
        if self.config["structure"]["edge"]["remove_edge_type_token"]:
            edge_structure_mapping = None
        # 5. stack node/edge/graph attr to nodes, so that total seq is Eulerian path len
        stack_func = (
            stack_node_edge_graph_attr_to_node
            if self.stack_method == "short"
            else stack_attr_to_node_and_edge
        )
        ls_tokens, ls_embed, ls_raw_node_idx = stack_func(
            self,
            path,
            node_structure_mapping,
            edge_structure_mapping,
            node_semantics_mapping,
            edge_semantics_mapping,
            graph_semantics_mapping,
        )
        # 6. add special tokens, e.g., eos
        ls_tokens = self.add_eos_token(ls_tokens) if self.add_eos else ls_tokens
        ls_embed = add_eos_embed(ls_embed) if self.add_eos else ls_embed
        ls_raw_node_idx.append(-1) if self.add_eos else None
        token_compontens = self.get_token_components(ls_tokens)
        ls_labels = ls_tokens[1:] + [[self.get_eos_token()] * token_compontens]
        # 6.1 enable nx func to enhance structure understanding
        # TODO: implement it
        # 6.2 enable instruction tuning to enhance semantics understanding
        (
            ls_instruct_tokens,
            ls_instruct_labels,
        ) = instruct_tuning_utils.follow_instructions(
            graph,
            tokenization_config=self.config,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            gtokenizer=self,
        )
        ls_tokens.extend(ls_instruct_tokens) if len(ls_instruct_tokens) > 0 else None
        ls_labels.extend(ls_instruct_labels) if len(ls_instruct_labels) > 0 else None
        if ls_embed:
            assert (
                len(ls_instruct_tokens) == 0
            ), "NOT implemented when embed inputs is presented"
        return tokenizer_utils.TokenizationOutput(
            ls_tokens=ls_tokens,
            ls_labels=ls_labels,
            tgt_node_token=tgt_node_token,
            tgt_edge_src_token=tgt_edge_src_token,
            tgt_edge_dst_token=tgt_edge_dst_token,
            tgt_edge_attr_token=tgt_edge_attr_token,
            ls_embed=ls_embed,
            ls_raw_node_idx=ls_raw_node_idx,
        )

    def convert_tokens_to_ids(self, seq_tokens: List[List[str]], seq_labels: List[str]):
        # 7. map tokens to token-id
        seq_tokens_id = [
            [self.vocab_map[token] for token in feat_tokens]
            for feat_tokens in seq_tokens
        ]
        seq_labels_id = [
            [self.vocab_map[token] for token in feat_tokens]
            for feat_tokens in seq_labels
        ]
        # 8. add labels, attention mask, position_ids and etc
        in_dict = get_input_dict_from_seq_tokens_id(
            seq_tokens_id,
            seq_labels_id,
            set(),
            None,
            self.cmpe,
            self.cyclic_mpe,
        )
        # in_dict["labels"] = [token_ids[0] for token_ids in in_dict["labels"]]
        return in_dict

    def prepare_inputs_for_task(
        self,
        in_dict: Dict,
        graph: Data,
        token_res: tokenizer_utils.TokenizationOutput,
        is_training: Optional[bool] = None,
    ):
        tgt_node_token_id = self._map_tokens_to_ids(token_res.tgt_node_token)
        tgt_edge_src_token_id = self._map_tokens_to_ids(token_res.tgt_edge_src_token)
        tgt_edge_dst_token_id = self._map_tokens_to_ids(token_res.tgt_edge_dst_token)
        tgt_edge_attr_token_id = self._map_tokens_to_ids(token_res.tgt_edge_attr_token)
        if self.stack_method == "long":
            if self.add_eos:
                ids = in_dict["input_ids"][:-1]
                eos = in_dict["input_ids"][-1:]
            else:
                ids = in_dict["input_ids"]
                eos = []
            node_attr_dim = self.config["semantics"]["node"]["dim"]
            in_dict["input_ids"] = [
                tokenizer_utils._pad_stacked_targets(
                    i,
                    ls_token_ids,
                    node_attr_dim=node_attr_dim,
                    padding_val=self.pad_token_id,
                    eos_token_id=self.get_eos_token_id(),
                )
                for i, ls_token_ids in enumerate(ids)
            ] + eos
        in_dict = prepare_inputs_for_task(
            self.task_type,
            in_dict,
            graph=graph,
            eos_token_id=self.get_eos_token_id(),
            tgt_node_token_id=tgt_node_token_id,
            tgt_edge_src_token_id=tgt_edge_src_token_id,
            tgt_edge_dst_token_id=tgt_edge_dst_token_id,
            tgt_edge_attr_token_id=tgt_edge_attr_token_id,
            tgt_pos=token_res.tgt_pos,
            gsum_token_id=self.get_gsum_token_id(),
            gtokenizer=self,
            ls_len=token_res.ls_len,
            ls_raw_node_idx=token_res.ls_raw_node_idx,
            **self.kwargs,
        )
        return in_dict


def add_eos_embed(ls_embed):
    if ls_embed:
        ls = [0.0] * len(ls_embed[0])
        ls_embed.append(ls)
    return ls_embed


def stack_node_edge_graph_attr_to_node(
    gtokenizer: StackedGSTTokenizer,
    path: List[Tuple[int, int]],
    node_structure_mapping,
    edge_structure_mapping,
    node_semantics_mapping,
    edge_semantics_mapping,
    graph_semantics_mapping,
):
    ls_tokens = []  # For next/masked-token-prediction
    ls_embed = []  # Embed features as input only
    ls_raw_node_idx = []  # raw node-idx for 3D position labeling

    # 1. work on 1st node in the path
    # 1.1 For discrete feature as tokens
    if path:
        node, _ = path[0]
    else:  # For graph with single node, path == []
        node = 0
    edge = (-1, -1)
    ls = instruct_tuning_utils._get_all_node_feats(
        node,
        edge,
        node_structure_mapping=node_structure_mapping,
        edge_structure_mapping=edge_structure_mapping,
        node_semantics_mapping=node_semantics_mapping,
        edge_semantics_mapping=edge_semantics_mapping,
    )
    ls_tokens.append(ls)

    # 1.2 For embed features
    ls_e = instruct_tuning_utils._get_all_node_feats(
        node,
        edge,
        node_semantics_mapping=node_semantics_mapping,
        edge_semantics_mapping=edge_semantics_mapping,
        edge_semantics_default=gtokenizer.get_default_edge_embed(),
        attr_type="embed",
    )
    ls_embed.append(ls_e)

    # 1.3 For raw node-idx
    ls_raw_node_idx.append(node)

    # 2. work on subsequent edges & nodes in the path
    for edge in path:
        _, node = edge
        # 2.1 For discrete feature as tokens
        ls = instruct_tuning_utils._get_all_node_feats(
            node,
            edge,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            edge_semantics_default=gtokenizer.get_default_edge_attr(),
        )
        ls_tokens.append(ls)
        # 2.2 For embed feature
        ls_e = instruct_tuning_utils._get_all_node_feats(
            node,
            edge,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            edge_semantics_default=gtokenizer.get_default_edge_embed(),
            attr_type="embed",
        )
        ls_embed.append(ls_e)
        # 2.3 For raw node-idx
        ls_raw_node_idx.append(node)
    return ls_tokens, ls_embed, ls_raw_node_idx


def stack_attr_to_node_and_edge(
    gtokenizer: StackedGSTTokenizer,
    path: List[Tuple[int, int]],
    node_structure_mapping,
    edge_structure_mapping,
    node_semantics_mapping,
    edge_semantics_mapping,
    graph_semantics_mapping,
):
    ls_tokens = []  # For next/masked-token-prediction
    ls_embed = []  # Embed features as input only
    ls_raw_node_idx = []  # raw node-idx for 3D position labeling

    # 1. work on 1st node in the path
    # 1.1 For discrete feature as tokens
    if path:
        node, _ = path[0]
    else:  # For graph with single node, path == []
        node = 0
    edge = (-1, -1)
    ls = instruct_tuning_utils._get_all_node_feats(
        node,
        edge,
        node_structure_mapping=node_structure_mapping,
        edge_structure_mapping=edge_structure_mapping,
        node_semantics_mapping=node_semantics_mapping,
        edge_semantics_mapping=edge_semantics_mapping,
    )
    ls_tokens.append(ls)

    # 1.2 For embed features
    ls_e = instruct_tuning_utils._get_all_node_feats(
        node,
        edge,
        node_semantics_mapping=node_semantics_mapping,
        edge_semantics_mapping=edge_semantics_mapping,
        attr_type="embed",
    )
    ls_embed.append(ls_e)
    pad_embed = tuple([0] * len(ls_e))

    # 1.3 For raw node-idx
    ls_raw_node_idx.append(node)

    for edge in path:
        # 2. obtain ls-tokens/embeds/node_idx from `edge`
        # 2.1 discrete features
        node = -1
        ls = instruct_tuning_utils._get_all_node_feats(
            node,
            edge,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            node_semantics_default=gtokenizer.get_default_node_attr(),
            edge_semantics_default=gtokenizer.get_default_edge_attr(),
        )
        ls_tokens.append(ls)
        # 2.2 embed features
        ls_embed.append(list(pad_embed))
        # 2.3 raw node-idx
        ls_raw_node_idx.append(node)

        # 3. obtain ls-tokens/embeds/node_idx from `node`
        _, node = edge
        edge = (-1, -1)
        # 3.1 discrete fatures
        ls = instruct_tuning_utils._get_all_node_feats(
            node,
            edge,
            node_structure_mapping=node_structure_mapping,
            edge_structure_mapping=edge_structure_mapping,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            node_semantics_default=gtokenizer.get_default_node_attr(),
            edge_semantics_default=gtokenizer.get_default_edge_attr(),
        )
        ls_tokens.append(ls)
        # 3.2 embed features
        ls_e = instruct_tuning_utils._get_all_node_feats(
            node,
            edge,
            node_semantics_mapping=node_semantics_mapping,
            edge_semantics_mapping=edge_semantics_mapping,
            attr_type="embed",
        )
        ls_embed.append(ls_e)
        # 3.3 raw node-idx
        ls_raw_node_idx.append(node)
    return ls_tokens, ls_embed, ls_raw_node_idx


def get_default_semantics_attr_mapping(graph: Data, config: Dict, node_or_edge: str):
    # input: path
    # output: a mapping of each node/edge to its attr, and each node to its global-idx if exists
    assert node_or_edge in {"node", "edge", "graph"}

    discrete_attr = config["semantics"][node_or_edge]["discrete"]
    share_vocab = config["semantics"][node_or_edge].get("share_vocab", False)
    world_identifier = config["attr_world_identifier"]
    ls_tokens = []
    if discrete_attr is not None:
        assert (
            len(graph[discrete_attr].shape) == 2
        ), f"{len(graph[discrete_attr].shape)} != 2"
        dim = graph[discrete_attr].shape[1]
        raw_attr = torch.zeros((1, dim)).to(graph[discrete_attr])
        # above is in case of graph[discrete_attr].shape[0] == 0, usually for single node, i.e., 0 edges
        raw_attr = raw_attr[0].numpy().astype(str)
        ls_tokens = _tokenize_discrete_attr(
            raw_attr,
            world_identifier,
            node_or_edge,
            remove_val=True,
            share_vocab=share_vocab,
        )
    return ls_tokens


def get_default_semantics_embed_mapping(graph: Data, config: Dict, node_or_edge: str):
    # input: path
    # output: a mapping of each node/edge to its attr, and each node to its global-idx if exists
    assert node_or_edge in {"node", "edge", "graph"}

    embed_attr = config["semantics"][node_or_edge].get("embed", None)
    default_embed = None
    if embed_attr is not None:
        assert len(graph[embed_attr].shape) == 2, f"{len(graph[embed_attr].shape)} != 2"
        dim = graph[embed_attr].shape[1]
        raw_attr = torch.zeros((1, dim)).to(graph[embed_attr])
        # above is in case of graph[embed_attr].shape[0] == 0, usually for single node, i.e., 0 edges
        default_embed = np.zeros_like(raw_attr[0].numpy()).tolist()
    return default_embed
