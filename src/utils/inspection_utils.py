import os
import numpy as np
from typing import Dict, Union
from pprint import pformat
from functools import partial
import torch
from torch_geometric.data import Data
from torch.utils.data import IterableDataset, Dataset

pformat = partial(pformat, compact=False, width=400)


def print_params(**kwargs):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        content = (
            f"Training Graph-GPT model with params:\n"
            f"output_dir: {kwargs['output_dir']}\n"
            f"raw_pretrain_cpt: {kwargs.get('raw_pretrain_cpt', None)}\n"
            f"pretrain_cpt: {kwargs.get('pretrain_cpt', None)}\n"
            f"data_dir: {kwargs['data_dir']}\n"
            f"dataset_name: {kwargs['dataset_name']}\n"
            f"odps_table: {kwargs['tables']}\n"
            f"with_prob: {kwargs.get('with_prob', 0)}\n"
            f"tokenization_config: {kwargs['tokenization_config']}\n"
            f"deepspeed_config: {kwargs.get('deepspeed_config', None)}\n"
            f"optimization_config: {kwargs.get('optimization_config', None)}\n"
            f"world_size: {kwargs.get('world_size', None)}\n"
            f"attr_assignment: {kwargs.get('attr_assignment', '')}\n"
            f"attr_shuffle: {kwargs.get('attr_shuffle', '')}\n"
            f"attr_mask_ratio: {kwargs.get('attr_mask_ratio', '')}\n"
            f"ignored_off: {kwargs.get('ignored_off', '')}\n"
            f"add_eos: {kwargs.get('add_eos', True)}\n"
            f"total_tokens: {kwargs.get('total_tokens', None)}\n"
            f"warmup_tokens: {kwargs.get('warmup_tokens', None)}\n"
            f"epochs: {kwargs.get('epochs', None)}\n"
            f"warmup_epochs: {kwargs.get('warmup_epochs', None)}\n"
            f"batch_size: {kwargs['batch_size']}\n"
            f"pad_to_multiple_of: {kwargs.get('pad_to_multiple_of', None)}\n"
            f"pack_tokens: {kwargs.get('pack_tokens', 0)}\n"
            f"lr: {kwargs['lr']}\n"
            f"min_lr: {kwargs.get('min_lr', None)}\n"
            f"betas: {kwargs['betas']}\n"
            f"eps: {kwargs['eps']}\n"
            f"weight_decay: {kwargs['weight_decay']}\n"
            f"max_grad_norm: {kwargs.get('max_grad_norm', None)}\n"
            f"logging_steps: {kwargs.get('logging_steps', None)}\n"
            f"freeze: {kwargs.get('freeze', 0)}\n"
            f"samples_per_eval: {kwargs.get('samples_per_eval', None)}\n"
            f"eval_steps: {kwargs.get('eval_steps', None)}\n"
            f"k_samplers: {kwargs.get('k_samplers', None)}\n"
            f"use_deepspeed: {kwargs.get('use_deepspeed', False)}\n"
            f"gradient_accumulation_steps: {kwargs.get('gradient_accumulation_steps', None)}\n"
            f"model_type: {kwargs.get('model_type', 'graphgpt')}\n"
            f"model_config: {kwargs.get('model_config', '')}\n"
            f"vocab_size: {kwargs['vocab_size']}\n"
            f"hidden_size: {kwargs['hidden_size']}\n"
            f"intermediate_size: {kwargs['intermediate_size']}\n"
            f"num_attention_heads: {kwargs['num_attention_heads']}\n"
            f"num_hidden_layers: {kwargs['num_hidden_layers']}\n"
            f"hidden_act: {kwargs['hidden_act']}\n"
            f"max_position_embeddings: {kwargs['max_position_embeddings']}\n"
            f"initializer_range: {kwargs['initializer_range']}\n"
            f"tie_word_embeddings: {kwargs.get('tie_word_embeddings', False)}\n"
            f"add_cls_token: {kwargs.get('add_cls_token', False)}\n"
            f"problem_type: {kwargs.get('problem_type', None)}\n"
            f"causal_attention: {kwargs.get('causal_attention', '')}\n"
            f"loss_type: {kwargs.get('loss_type', None)}\n"
            f"ntp_ratio: {kwargs.get('ntp_ratio', None)}\n"
            f"task_ratio: {kwargs.get('task_ratio', None)}\n"
            f"uni_loss_ratio: {kwargs.get('uni_loss_ratio', None)}\n"
            f"bi_loss_ratio: {kwargs.get('bi_loss_ratio', None)}\n"
            f"num_labels: {kwargs.get('num_labels', None)}\n"
            f"mlp: {kwargs.get('mlp', None)}\n"
            f"dropout: {kwargs.get('dropout', 0)}\n"
            f"task_level: {kwargs.get('task_level', None)}\n"
            f"pretrain_wgt: {kwargs.get('pretrain_wgt', None)}\n"
            f"tables: {kwargs.get('tables', None)}\n"
            f"outputs: {kwargs.get('outputs', None)}\n"
            f"samples_per_saving: {kwargs.get('samples_per_saving', None)}\n"
            f"steps_per_saving: {kwargs.get('steps_per_saving', None)}\n"
            f"do_valid: {kwargs.get('do_valid', None)}\n"
            f"do_test: {kwargs.get('do_test', None)}\n"
            f"eval_only: {kwargs.get('eval_only', None)}\n"
            f"seed: {kwargs.get('seed', None)}\n"
            f"gpu: {kwargs.get('gpu_name', None)}\n"
        )
        print(content)
        return content


def print_trainable_parameters(model):
    """
    copied from `peft/peft_model.py`
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def inspect_nodes(dataset):
    ls_nodes_cnt = []
    for paths, _ in dataset:
        path = paths[0]
        cnt_nodes = max([tgt for src, tgt, scope in path])
        ls_nodes_cnt.append(cnt_nodes)
    cnt_max = max(ls_nodes_cnt)
    cnt_95 = np.percentile(ls_nodes_cnt, 95)
    cnt_90 = np.percentile(ls_nodes_cnt, 90)
    cnt_50 = np.percentile(ls_nodes_cnt, 50)
    cnt_min = min(ls_nodes_cnt)
    cnt_mean = np.mean(ls_nodes_cnt)
    print(
        f"max nodes cnt: {cnt_max}, 95 percentile cnt: {cnt_95}, 90 percentile cnt: {cnt_90}, median cnt: {cnt_50}, min cnt: {cnt_min}, mean cnt: {cnt_mean}"
    )
    return cnt_max


def inspect_sequences(dataset):
    ls_max_len = []
    for paths, _ in dataset:
        max_len = max([len(path) for path in paths])
        ls_max_len.append(max_len)
    len_max = max(ls_max_len)
    len_99 = np.percentile(ls_max_len, 99)
    len_98 = np.percentile(ls_max_len, 98)
    len_97 = np.percentile(ls_max_len, 97)
    len_96 = np.percentile(ls_max_len, 96)
    len_95 = np.percentile(ls_max_len, 95)
    len_90 = np.percentile(ls_max_len, 90)
    len_50 = np.percentile(ls_max_len, 50)
    len_min = min(ls_max_len)
    len_mean = np.mean(ls_max_len)
    print(
        f"max path len: {len_max}, 99/98/97/96/95 percentile len: {len_99}/{len_98}/{len_97}/{len_96}/{len_95}, 90 percentile len: {len_90}, median len: {len_50}, min len: {len_min}, mean len: {len_mean}"
    )


def inspect_tokenization_results(
    dataset: Union[Dataset, IterableDataset], gtokenizer, idx: int = None
):
    if isinstance(dataset, IterableDataset):
        data = next(iter(dataset))
        if isinstance(data, tuple):
            _, data = data
        # return
    else:
        idx = 0 if idx is None else idx
        idx = dataset.sampler[idx] if hasattr(dataset, "sampler") else idx
        print(f"\nInspecting graph of index {idx}")
        idx2, data = dataset[idx]
        if idx != idx2:
            print(f"[Warning]Local idx {idx2} NOT equal Global idx {idx}")
    ls_embed = []
    if isinstance(data, Data):
        graph = data
        print(f"Inspecting tokenization results!\nTokenize graph:\n{data}")
        token_res = gtokenizer.tokenize(graph)
        print(
            f"\nTokens:\n{pformat(token_res.ls_tokens)}\nLabels:\n{pformat(token_res.ls_labels)}\nembed:{np.array(token_res.ls_embed)}\n"
        )
        tokens, labels, ls_embed, ls_len = (
            gtokenizer.pack_token_seq(token_res, idx)
            if gtokenizer.mpe is not None
            else (
                token_res.ls_tokens,
                token_res.ls_labels,
                token_res.ls_embed,
                [len(token_res.ls_tokens)],
            )
        )
        print(
            f"Packed Tokens:\n{pformat(tokens)}\nPacked Labels:\n{pformat(labels)}\nPacked embed:\n{np.array(ls_embed).shape}\n{np.array(ls_embed)}\nPacked len:\n{pformat(ls_len)}"
        ) if gtokenizer.mpe is not None else None
        in_dict = gtokenizer.convert_tokens_to_ids(tokens, labels)
        if ls_embed:  # for pretty print purpose ONLY
            in_dict["embed"] = np.array(ls_embed)
        print(f"Tokenized results:\n{pformat(in_dict)}\n")
        if ls_embed:
            in_dict["embed"] = ls_embed
        token_res.ls_tokens = tokens
        token_res.ls_labels = labels
        token_res.ls_embed = ls_embed
        token_res.ls_len = ls_len
        inputs = gtokenizer.prepare_inputs_for_task(
            in_dict,
            graph,
            token_res=token_res,
        )
    elif isinstance(data, Dict):
        inputs = data
    else:
        raise ValueError(f"Type {type(data)} of data {data} is NOT implemented yet!")
    if ls_embed:  # for pretty print purpose ONLY
        inputs["embed"] = np.array(ls_embed)
    print(f"Inputs for model:\n{pformat(inputs)}\n")
    gtokenizer.set_eos_idx(inputs["input_ids"])


def inspect_attr(attr, attr_name):
    dict_names = {
        "atom": [
            "possible_atomic_num_list",
            "possible_chirality_list",
            "possible_degree_list",
            "possible_formal_charge_list",
            "possible_numH_list",
            "possible_number_radical_e_list",
            "possible_hybridization_list",
            "possible_is_aromatic_list",
            "possible_is_in_ring_list",
        ],
        "bond": ["0", "1", "2"],
    }
    if attr_name in dict_names.keys():
        attr_name = dict_names[attr_name]
    assert attr.shape[1] == len(attr_name)
    for i in range(attr.shape[1]):
        unique_values, counts = torch.unique(attr[:, i], return_counts=True)
        for val, cnt in zip(unique_values, counts):
            print(f"{attr_name[i]} {i}:{val} => cnt: {cnt}")
