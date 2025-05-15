from typing import Dict, Optional, Tuple
import math
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.optim.lr_scheduler import (
    CyclicLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)
from . import control_flow

_ds_scheduler = control_flow.Register()
set_ds_scheduler = _ds_scheduler.build  # return func results
get_ds_scheduler_func = _ds_scheduler.get  # return the func

DS_SCHEDULER_LS = _ds_scheduler._register_map.keys()

_py_scheduler = control_flow.Register()
set_py_scheduler = _py_scheduler.build  # return func results


def auc_loss(y_pred, y_true, num_neg: int = 1):
    """
    :param y_pred: 1-D, [N]
    :param y_true: 1-D, [N]
    :param num_neg: int
    :return:
    """
    # y_pred is logits of 1-D
    pos_mask = y_true.bool()
    y_pred_pos = y_pred[pos_mask]

    neg_mask = (1 - y_true).bool()
    y_pred_neg = y_pred[neg_mask]

    cnt_negs = y_pred_pos.shape[0] * num_neg
    neg_samples = y_pred_neg.shape[0]
    idx = (
        torch.randperm(cnt_negs, dtype=torch.int64, device=y_pred.device) % neg_samples
    )
    y_pred_neg = y_pred_neg[idx]  # [neg_samples] -> [cnt_negs]
    return _auc_loss(y_pred_pos, y_pred_neg, num_neg)


def _auc_loss(pos_out, neg_out, num_neg):
    # refer: https://github.com/skepsun/Adaptive-Graph-Diffusion-Networks/blob/master/ogbl_no_sampling/src/loss.py
    # useful for OGBL-DDI dataset
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return torch.square(1 - (pos_out - neg_out)).mean()


def get_neg_ratio(sampling_config: Dict):
    if sampling_config is None:
        return None
    ls = []
    for key, each_sampling in sampling_config.items():
        if each_sampling["valid"] == 1:
            neg_ratio = each_sampling.get("neg_ratio", 1)
            ls.append(neg_ratio)
    print(f"neg_ratio is {ls}, and the max {max(ls)} will be returned!")
    return max(ls)


def update_logits_prior(prior: np.ndarray, labels: torch.Tensor, shape: Tuple[int]):
    # https://spaces.ac.cn/archives/7615
    labels = labels.detach().cpu().numpy()
    i, v = np.unique(labels, return_counts=True)
    iv = [(each_i, each_v) for each_i, each_v in zip(i, v) if each_i != -100]
    ii = [each_i for each_i, each_v in iv]
    vv = [each_v for each_i, each_v in iv]
    new_prior = np.zeros(shape, dtype=np.int64)
    new_prior[ii] = vv
    return prior + new_prior


def convert_prior_cnt_to_logits(prior: Optional[np.ndarray], extra_dims, device):
    if prior is None:
        return None
    prior_f = prior + 1e-8
    prior_f = np.log(prior_f) - np.log(prior_f.sum())
    logits_adjust = prior_f.reshape([1] * extra_dims + [-1])
    return torch.tensor(logits_adjust).float().to(device)


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def _dist_infonce(
    output_embeds: torch.Tensor,
    tb_output_embeds: torch.Tensor,
    *,
    temperature: float = 20.0,
    world_size: int = None,
):
    # refer to Alibaba internal doc: https://aliyuque.antfin.com/amils0/interest/zcnufr
    # output_embeds: l2-normalized, [bz, dim]
    # tb_output_embeds: l2-normalized, [bz, dim]
    # whether to use barrier?
    # torch.distributed.barrier()
    if world_size != 1:
        output_embeds = GatherLayer.apply(output_embeds)
        tb_output_embeds = GatherLayer.apply(tb_output_embeds)

        output_embeds = torch.cat(output_embeds, dim=0)
        tb_output_embeds = torch.cat(tb_output_embeds, dim=0)

    # print(f"output_embeds: {output_embeds.shape}, tb_output_embeds: {tb_output_embeds.shape}")
    scores, batch_size = cos_sim(output_embeds, tb_output_embeds, temperature)
    # print(f"scores: {scores.shape}\n{scores}")
    _labels = torch.arange(batch_size, dtype=torch.long, device=scores.device)
    # print(f"_labels: {_labels.shape}\n{_labels}")

    loss_fct = nn.CrossEntropyLoss()
    # if cos_sim with `expand==True`, then no need below symmetric loss calculation
    loss = 0.5 * (
        loss_fct(scores.float(), _labels) + loss_fct(scores.T.float(), _labels)
    )
    return loss


def cos_sim(
    left_embeds: torch.Tensor,
    right_embeds: torch.Tensor,
    temperature: float,
    expand: bool = True,
):
    assert (
        left_embeds.size() == right_embeds.size()
    ), f"{left_embeds.size()} != {right_embeds.size()}"
    if expand:
        dim = left_embeds.size(1)
        left_new = torch.hstack([left_embeds, right_embeds]).reshape((-1, dim))
        right_new = torch.hstack([right_embeds, left_embeds]).reshape((-1, dim))
        scores = torch.mm(left_new, right_new.T) * temperature

        bz = left_new.size(0)
        dtype = left_new.dtype
        device = left_new.device
        # BELOW mask the entries with score == 1*temperature
        x_idx = torch.arange(bz, dtype=torch.long, device=device)
        y_idx = x_idx + 1 - (x_idx % 2) * 2
        # print(f"[DEBUG] before mask: {scores[x_idx, y_idx]}")
        scores[x_idx, y_idx] = torch.finfo(dtype).min
        # print(f"[DEBUG] after mask: {scores[x_idx, y_idx]}\nscores:\n{scores}")
    else:
        scores = torch.mm(left_embeds, right_embeds.T) * temperature
        bz = left_embeds.size(0)
    return scores, bz


@_ds_scheduler("WarmupLR")
def _ds_warmup(
    params_dict: Dict,
    *,
    warmup_max_lr: float = 3e-4,
    warmup_min_lr: float = 0,
    warmup_num_steps: int = 2000,
    **kwargs,
):
    params_dict["warmup_max_lr"] = warmup_max_lr
    params_dict["warmup_min_lr"] = warmup_min_lr
    params_dict["warmup_num_steps"] = warmup_num_steps
    params_dict.pop("total_num_steps", None)
    return params_dict


@_ds_scheduler("WarmupDecayLR")
def _ds_warmup_decay(
    params_dict: Dict,
    *,
    warmup_max_lr: float = 3e-4,
    warmup_min_lr: float = 0,
    warmup_num_steps: int = 2000,
    total_num_steps: int = None,
    **kwargs,
):
    params_dict["warmup_max_lr"] = warmup_max_lr
    params_dict["warmup_min_lr"] = warmup_min_lr
    params_dict["warmup_num_steps"] = warmup_num_steps
    params_dict["total_num_steps"] = total_num_steps
    return params_dict


@_ds_scheduler("OneCycle")
def _ds_one_cycle(
    params_dict: Dict,
    *,
    cycle_min_lr: float = 1e-6,
    cycle_max_lr: float = 3e-4,
    cycle_first_step_size: int = 2000,
    **kwargs,
):
    params_dict["cycle_min_lr"] = cycle_min_lr
    params_dict["cycle_max_lr"] = cycle_max_lr
    params_dict["cycle_first_step_size"] = cycle_first_step_size
    return params_dict


@_ds_scheduler("LRRangeTest")
def _ds_lr_range_test(
    params_dict: Dict,
    *,
    lr_range_test_min_lr: float = 1e-6,
    lr_range_test_step_size: int = 2000,
    lr_range_test_step_rate: float = 1,
    **kwargs,
):
    params_dict["lr_range_test_min_lr"] = lr_range_test_min_lr
    params_dict["lr_range_test_step_size"] = lr_range_test_step_size
    params_dict["lr_range_test_step_rate"] = lr_range_test_step_rate
    return params_dict


@_py_scheduler("CyclicLR")
def _py_cyclic(
    ds_config: Dict,
    *,
    base_lr: float = 1e-9,
    max_lr: float = 3e-4,
    step_size_up: int = 2000,
    last_step_index: int = -1,
    **kwargs,
):
    ds_config["scheduler"]["params"] = {
        "base_lr": base_lr,
        "max_lr": max_lr,
        "step_size_up": step_size_up,
        "last_epoch": last_step_index,
    }
    scheduler_conf = ds_config.pop("scheduler")
    print(f"Pop scheduler in ds_config: {scheduler_conf}")

    def func(optimizer):
        return CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            cycle_momentum=False,
            last_epoch=last_step_index,
        )

    # TODO: re-implement CyclicLR to support cyclic adam's 1st momentum
    return func, {"scheduler": scheduler_conf}


@_py_scheduler("CosineAnnealingLR")
def _py_cosine_annealing(
    ds_config: Dict,
    *,
    T_max: int = 100000,  # Maximum number of iterations
    eta_min: float = 0,  # Minimum learning rate. Default: 0
    last_step_index: int = -1,
    **kwargs,
):
    ds_config["scheduler"]["params"] = {
        "T_max": T_max,
        "eta_min": eta_min,
        "last_epoch": last_step_index,
    }
    scheduler_conf = ds_config.pop("scheduler")
    print(f"Pop scheduler in ds_config: {scheduler_conf}")

    def func(optimizer):
        return CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_step_index
        )

    return func, {"scheduler": scheduler_conf}


@_py_scheduler("CosineAnnealingWarmRestarts")
def _py_cosine_annealing_wr(
    ds_config: Dict,
    *,
    T_0: int = 100000,  # Number of iterations for the first restart.
    T_mult: int = 1,  # A factor increases T_i after a restart. Default: 1.
    eta_min: float = 0,  # Minimum learning rate. Default: 0.
    last_step_index: int = -1,
    **kwargs,
):
    ds_config["scheduler"]["params"] = {
        "T_0": T_0,
        "T_mult": T_mult,
        "eta_min": eta_min,
        "last_epoch": last_step_index,
    }
    scheduler_conf = ds_config.pop("scheduler")
    print(f"Pop scheduler in ds_config: {scheduler_conf}")

    def func(optimizer):
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_step_index,
        )

    return func, {"scheduler": scheduler_conf}


@_py_scheduler("OneCycleLR")
def _py_one_cycle(
    ds_config: Dict,
    *,
    max_lr: float = 3e-4,
    min_lr: float = 0,
    total_steps: int = 10000,
    pct_start: float = 0.1,  # The percentage of the cycle (in number of steps) spent increasing the learning rate.
    last_step_index: int = -1,
    **kwargs,
):
    div_factor = 25
    final_div_factor = 1e4
    # default from `https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR`
    initial_lr = max_lr / div_factor
    if min_lr > 0:
        final_div_factor = initial_lr / min_lr
    else:
        min_lr = initial_lr / final_div_factor
    ds_config["scheduler"]["params"] = {
        "max_lr": max_lr,
        "min_lr": min_lr,
        "div_factor": div_factor,
        "final_div_factor": final_div_factor,
        "total_steps": total_steps,
        "pct_start": pct_start,
        "last_epoch": last_step_index,
    }
    scheduler_conf = ds_config.pop("scheduler")
    print(f"Pop scheduler in ds_config: {scheduler_conf}")

    def func(optimizer):
        lrs = [dict_.get("lr", max_lr) for dict_ in optimizer.param_groups]
        return OneCycleLR(
            optimizer,
            max_lr=lrs,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            last_epoch=last_step_index,
        )

    return func, {"scheduler": scheduler_conf}


def get_layerwise_param_groups(model, base_lr, lr_decay: float = 0.95):
    print("utilizing layerwise lr v1!")
    params_groups = []
    params_groups.append({"params": model.model.embed_tokens.parameters()})
    if hasattr(model, "stacked_feat_agg"):
        params_groups.append({"params": model.stacked_feat_agg.parameters()})
    for layer in model.model.layers:
        params_groups.append({"params": layer.parameters()})
    params_groups.append({"params": model.model.norm.parameters()})
    if hasattr(model, "lm_head"):
        params_groups.append({"params": model.lm_head.parameters()})
    if hasattr(model, "score"):
        params_groups.append({"params": model.score.parameters()})
    [
        dict_.update({"lr": math.pow(lr_decay, i) * base_lr})
        for i, dict_ in enumerate(params_groups[::-1])
    ]
    return params_groups


def get_layerwise_param_groups_v2(model, base_lr, lr_decay: float = 0.95):
    param_groups = [
        {"params": list(module.parameters()), "lr": base_lr * (lr_decay**depth)}
        for depth, module in enumerate(reversed(list(model.children())))
    ]
    return param_groups


def get_layerwise_param_groups_v3(model, base_lr, lr_decay: float = 0.5):
    param_groups = [{"params": []}, {"params": []}]
    param_groups[0]["params"].extend(list(model.model.parameters()))
    if hasattr(model, "stacked_feat_agg"):
        param_groups[0]["params"].extend(list(model.stacked_feat_agg.parameters()))

    if hasattr(model, "lm_head"):
        param_groups[1]["params"].extend(list(model.lm_head.parameters()))
    if hasattr(model, "score"):
        param_groups[1]["params"].extend(list(model.score.parameters()))
    [
        dict_.update({"lr": math.pow(lr_decay, i) * base_lr})
        for i, dict_ in enumerate(param_groups[::-1])
    ]
    return param_groups
