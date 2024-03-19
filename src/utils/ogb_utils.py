from typing import Dict
from torch import Tensor
from . import control_flow

_eval = control_flow.Register()
evaluate_ogb = _eval.build  # return func results
get_ogb_evaluator = _eval.get  # return the func


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
