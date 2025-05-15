from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import Tensor
from . import control_flow

_eval = control_flow.Register()
evaluate_ogb = _eval.build  # return func results
get_ogb_evaluator = _eval.get  # return the func


def _eval_rocauc(y_true, y_pred):
    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(
                roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            )

    if len(rocauc_list) == 0:
        raise RuntimeError(
            "No positively labeled data available. Cannot compute ROC-AUC."
        )

    return {"rocauc": sum(rocauc_list) / len(rocauc_list)}


@_eval("reddit_threads")
def _eval_reddit_threads(input_dict: Dict[str, Tensor]):
    # In most cases, input_dict is
    # input_dict = {"y_true": y_true, "y_pred": y_pred}
    input_dict = {k: v.reshape((-1, 1)) for k, v in input_dict.items()}
    y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    result_dict = _eval_rocauc(y_true, y_pred)
    result_dict["ema_rocauc"] = result_dict.pop("rocauc")
    return result_dict


@_eval("ogbn-arxiv")
def _eval_ogbn_arxiv(input_dict: Dict[str, Tensor]):
    from ogb.nodeproppred import Evaluator

    evaluator = Evaluator(name="ogbn-arxiv")
    # In most cases, input_dict is
    # input_dict = {"y_true": y_true, "y_pred": y_pred}
    input_dict = {k: v.reshape((-1, 1)) for k, v in input_dict.items()}
    result_dict = evaluator.eval(input_dict)
    result_dict["ema_acc"] = result_dict.pop("acc")
    return result_dict


@_eval("ogbn-products")
def _eval_ogbn_products(input_dict: Dict[str, Tensor]):
    from ogb.nodeproppred import Evaluator

    evaluator = Evaluator(name="ogbn-products")
    # In most cases, input_dict is
    # input_dict = {"y_true": y_true, "y_pred": y_pred}
    input_dict = {k: v.reshape((-1, 1)) for k, v in input_dict.items()}
    result_dict = evaluator.eval(input_dict)
    result_dict["ema_acc"] = result_dict.pop("acc")
    return result_dict


@_eval("ogbn-proteins")
def _eval_ogbn_proteins(input_dict: Dict[str, Tensor]):
    from ogb.nodeproppred import Evaluator

    evaluator = Evaluator(name="ogbn-proteins")
    # In most cases, input_dict is
    # input_dict = {"y_true": y_true, "y_pred": y_pred}
    result_dict = evaluator.eval(input_dict)
    return result_dict


@_eval("ogbl-ppa")
def _eval_ogbl_ppa(input_dict: Dict[str, Tensor]):
    from ogb.linkproppred import Evaluator

    evaluator = Evaluator(name="ogbl-ppa")
    y_pred_pos, y_pred_neg = _reformat_pred_for_hr_eval(input_dict)
    result_dict = evaluator.eval({"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg})
    return result_dict


@_eval("ogbl-citation2")
def _eval_ogbl_citation2(input_dict: Dict[str, Tensor]):
    from ogb.linkproppred import Evaluator

    evaluator = Evaluator(name="ogbl-citation2")
    y_pred_pos, y_pred_neg = _reformat_pred_for_mrr_eval(input_dict, cnt_neg=1000)
    result_dict = evaluator.eval({"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg})
    print({k: v.shape for k, v in result_dict.items()})
    result_dict = {k: torch.mean(v.float()).item() for k, v in result_dict.items()}
    result_dict["ema_mrr_list"] = result_dict.pop("mrr_list")
    return result_dict


@_eval("ogbl-wikikg2")
def _eval_ogbl_wikikg2(input_dict: Dict[str, Tensor]):
    from ogb.linkproppred import Evaluator

    evaluator = Evaluator(name="ogbl-wikikg2")
    y_pred_pos, y_pred_neg = _reformat_pred_for_mrr_eval(input_dict, cnt_neg=1000)
    # eval below refer to: https://github.com/snap-stanford/ogb/blob/f631af76359c9687b2fe60905557bbb241916258/examples/linkproppred/wikikg2/model.py#L328
    y_pred_neg_head_batch = y_pred_neg[:, range(0, y_pred_neg.shape[1], 2)]
    y_pred_neg_tail_batch = y_pred_neg[:, range(1, y_pred_neg.shape[1], 2)]
    result_dict_head_batch = evaluator.eval(
        {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg_head_batch}
    )
    result_dict_tail_batch = evaluator.eval(
        {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg_tail_batch}
    )
    result_dict = {}
    for k in result_dict_head_batch.keys():
        result_dict[k] = torch.cat(
            [result_dict_head_batch[k], result_dict_tail_batch[k]]
        )
    print({k: v.shape for k, v in result_dict.items()})
    result_dict = {k: torch.mean(v.float()).item() for k, v in result_dict.items()}
    result_dict["ema_mrr_list"] = result_dict.pop("mrr_list")
    return result_dict


@_eval("ogbl-ddi")
def _eval_ogbl_ddi(input_dict: Dict[str, Tensor]):
    from ogb.linkproppred import Evaluator

    evaluator = Evaluator(name="ogbl-ddi")
    y_pred_pos, y_pred_neg = _reformat_pred_for_hr_eval(input_dict)
    result_dict = evaluator.eval({"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg})
    return result_dict


def _reformat_pred_for_hr_eval(input_dict):
    # In most cases, input_dict is
    # input_dict = {"y_true": y_true, "y_pred": y_pred}
    y_true = input_dict["y_true"]
    y_pred = input_dict["y_pred"]

    pos_mask = y_true.bool()
    y_pred_pos = y_pred[pos_mask]

    neg_mask = (1 - y_true).bool()
    y_pred_neg = y_pred[neg_mask]
    return y_pred_pos, y_pred_neg


def _reformat_pred_for_mrr_eval(input_dict: Dict, cnt_neg: int):
    # mainly for ogbl-citation2 & ogbl-wikikg2
    idx, indices = torch.sort(input_dict["idx"])
    y_true = input_dict["y_true"][indices]
    y_pred = input_dict["y_pred"][indices]
    assert idx.max().item() + 1 == len(idx), f"{idx.max().item()}+1 != {len(idx)}"

    pos_mask = y_true.bool()
    y_pred_pos = y_pred[pos_mask]
    y_pred_neg = y_pred[~pos_mask]
    y_pred_neg = y_pred_neg.reshape((-1, cnt_neg))

    assert (
        y_pred_pos.shape[0] == y_pred_neg.shape[0]
    ), f"{y_pred_pos.shape[0]} != {y_pred_neg.shape[0]}"
    return y_pred_pos, y_pred_neg


@_eval("ogbg-molhiv")
def _eval_ogbg_molhiv(input_dict: Dict[str, Tensor]):
    from ogb.graphproppred import Evaluator

    evaluator = Evaluator(name="ogbg-molhiv")
    # In most cases, input_dict is
    # input_dict = {"y_true": y_true, "y_pred": y_pred}
    input_dict = {
        k: v.reshape((-1, 1)) if v.ndim != 2 else v for k, v in input_dict.items()
    }
    result_dict = evaluator.eval(input_dict)
    return result_dict


@_eval("ogbg-molpcba")
def _eval_ogbg_molpcba(input_dict: Dict[str, Tensor]):
    from ogb.graphproppred import Evaluator

    evaluator = Evaluator(name="ogbg-molpcba")
    # In most cases, input_dict is
    # input_dict = {"y_true": y_true, "y_pred": y_pred}
    result_dict = evaluator.eval(input_dict)
    return result_dict


@_eval("PCQM4Mv2")
def _eval_pcqm4mv2(input_dict: Dict[str, Tensor]):
    from ogb.lsc import PCQM4Mv2Evaluator

    evaluator = PCQM4Mv2Evaluator()
    result_dict = evaluator.eval(input_dict)
    return result_dict


def format_ogb_output_for_csv(eval_res, rnd: int = 6):
    if isinstance(eval_res, Dict):
        return ",".join(
            [",".join([str(k), str(round(v, rnd))]) for k, v in eval_res.items()]
        )
    else:
        return eval_res
