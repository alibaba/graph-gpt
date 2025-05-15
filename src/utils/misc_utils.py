import os
import time
import shutil
import random
import numpy as np
from datetime import datetime
import pandas as pd
from typing import List
from tqdm import tqdm
import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.trainer import (
    OPTIMIZER_NAME,
    OPTIMIZER_NAME_BIN,
    SCHEDULER_NAME,
    SCALER_NAME,
)

MODEL_NAME = "model.pt"


def delete_old_ckp(dir_ckp):
    if os.path.exists(dir_ckp):
        shutil.rmtree(dir_ckp, ignore_errors=True)


def get_latest_ckp(pretrain_cpt, eval_only=0):
    dirs_scan = [f.path for f in os.scandir(pretrain_cpt) if f.is_dir()]
    dirs_scan = [dir_.split("/")[-1] for dir_ in dirs_scan]
    dirs_scan = [dir_.split("_")[-1] for dir_ in dirs_scan]
    dirs_scan = [int(dir_) for dir_ in dirs_scan if dir_.isnumeric()]
    if dirs_scan:
        ep = min(dirs_scan) if eval_only else max(dirs_scan)
        ckp = os.path.join(pretrain_cpt, f"epoch_{ep}")
    else:
        ep = None
        ckp = pretrain_cpt
    return ckp, ep


def convert_dict_to_df(dict_):
    dict_ = {key: val.float().detach().cpu().numpy() for key, val in dict_.items()}
    return pd.DataFrame(dict_)


def save_pred_results(dict_, model_dir, name):
    df = convert_dict_to_df(dict_)
    fn = os.path.join(model_dir, f"{name}_results.csv")
    df.to_csv(fn, index=False)
    print(f"{name} results saved in {fn}!")


def save_ckp(
    output_dir, model, epoch, use_deepspeed, optimizer=None, lr_scheduler=None
):
    rank = int(os.environ.get("RANK", 0))
    if use_deepspeed:
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        model_dir = os.path.join(output_dir, f"epoch_{epoch}")
        model_dir_dp = os.path.join(model_dir, f"global_step{model.global_steps}")
        # below to avoid FileExistsError when saving ckp in Nebula
        if rank != 0:
            while not os.path.exists(model_dir_dp):
                print(f"waiting for model dir {model_dir_dp} to be created!")
                time.sleep(3)
            time.sleep(3)
        else:
            os.makedirs(model_dir_dp, exist_ok=True)
        print(f"Model dir {model_dir}\n{model_dir_dp}\ncreated, continue ...")
        # a). save model ckp, including optimizer stats. all processes must call this method
        model.save_checkpoint(model_dir)
        print(f"Model saved in {model_dir} using deepspeed API.")
        # b). delete old ckp
        if rank == 0:
            old_model_dir = os.path.join(output_dir, f"epoch_{epoch - 5}")
            delete_old_ckp(old_model_dir)
    else:
        if rank == 0:
            save_all(
                output_dir,
                model,
                epoch,
                save_model=True,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )


def save_ddp_ckp(output_dir, model, optimizer, lr_scheduler):
    assert isinstance(model, DDP), f"model type: {type(model)}"
    # 1. save model params
    # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#save-and-load-checkpoints
    fn_model = os.path.join(output_dir, MODEL_NAME)
    torch.save(model.state_dict(), fn_model)
    print(f"DDP Model saved in {fn_model} using torch API.")
    # 2. save optimizer stats
    if optimizer is not None:
        fn_optimizer = os.path.join(output_dir, OPTIMIZER_NAME)
        torch.save(optimizer.state_dict(), fn_optimizer)
        print(f"Optimizer saved in {fn_optimizer}")
    # 3. save scheduler
    if lr_scheduler is not None:
        fn_scheduler = os.path.join(output_dir, SCHEDULER_NAME)
        torch.save(lr_scheduler.state_dict(), fn_scheduler)
        print(f"Scheduler saved in {fn_scheduler}")


def save_all(
    output_dir,
    model,
    epoch,
    *,
    save_model: bool = True,
    ls_log=None,
    ls_result=None,
    ls_loss=None,
    tr_dict=None,
    val_dict=None,
    test_dict=None,
    world_size: int = 1,
    optimizer=None,
    lr_scheduler=None,
):
    model_dir = os.path.join(output_dir, f"epoch_{epoch}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    # a). save model ckp
    if save_model:
        save_ddp_ckp(model_dir, model, optimizer, lr_scheduler)
    # b). delete old ckp
    old_model_dir = os.path.join(output_dir, f"epoch_{epoch - 5*world_size}")
    delete_old_ckp(old_model_dir)
    # c). save train/eval loss/metrics logs
    if ls_log:
        fn_log = f"{output_dir}/log.csv"
        with open(fn_log, "w") as fp:
            fp.writelines(ls_log)
        print(f"log saved in {fn_log}!")
    # d). save eval/test metrics
    if ls_result:
        fn_result = f"{output_dir}/result.csv"
        with open(fn_result, "w") as fp:
            fp.writelines(ls_result)
        print(f"log saved in {fn_result}!")
    # e). save fine-tune training loss
    if ls_loss:
        fn_loss = f"{output_dir}/loss.csv"
        with open(fn_loss, "w") as fp:
            fp.writelines(ls_loss)
        print(f"training loss saved in {fn_loss}!")
    # f). save prediction results
    if tr_dict:
        save_pred_results(tr_dict, model_dir, "train")

    if val_dict:
        save_pred_results(val_dict, model_dir, "valid")

    if test_dict:
        save_pred_results(test_dict, model_dir, "test")


def _load_log(fn):
    with open(fn, "r") as fp:
        ls_log = fp.readlines()
    print(f"log loaded from {fn}!")
    return ls_log


def load_all(
    output_dir,
    *,
    load_log=True,
    load_result=False,
    load_loss=False,
):
    ls_log, ls_result, ls_loss = None, None, None
    # 1). load train/eval loss/metrics logs
    if load_log:
        fn_log = f"{output_dir}/log.csv"
        ls_log = _load_log(fn_log)
    # 2). load eval/test metrics
    if load_result:
        fn_result = f"{output_dir}/result.csv"
        ls_result = _load_log(fn_result)
    # 3). load fine-tune training loss
    if load_loss:
        fn_loss = f"{output_dir}/loss.csv"
        ls_loss = _load_log(fn_loss)
    return ls_log, ls_result, ls_loss


def load_ddp_ckp(
    output_dir,
    *,
    model=None,
    optimizer=None,
    lr_scheduler=None,
):
    if model is not None:
        assert isinstance(model, DDP)
        # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#save-and-load-checkpoints
        fn_model = os.path.join(output_dir, MODEL_NAME)
        model.load_state_dict(torch.load(fn_model))
        print(f"load DDP model params from {fn_model} using torch API")
    if optimizer is not None:
        fn_optimizer = os.path.join(output_dir, OPTIMIZER_NAME)
        optimizer.load_state_dict(torch.load(fn_optimizer))
        print(f"load optimizer from {fn_optimizer}")
    if lr_scheduler is not None:
        fn_scheduler = os.path.join(output_dir, SCHEDULER_NAME)
        lr_scheduler.load_state_dict(torch.load(fn_scheduler))
        print(f"load scheduler from {fn_scheduler}")


def estimate_tokens_per_sample(
    gtokenizer, dataset, sampler, mpe, world_size, tot_samples: int = 10000
):
    num_samples = tot_samples // world_size
    if not isinstance(dataset, IterableDataset):
        num_samples = min(num_samples, len(sampler))
        sampler = random.sample(sampler, num_samples)
        ls_seq = [
            len(gtokenizer.tokenize(dataset[idx][-1]).ls_tokens)
            for idx in tqdm(sampler)
        ]
    else:
        ls_seq = []
        print(f"Estimating tokens_per_sample ...")
        for i, ele in enumerate(iter(dataset)):
            if i >= num_samples:
                break
            ls_seq.append(len(gtokenizer.tokenize(ele[-1]).ls_tokens))
    q = torch.tensor(np.minimum(np.array(ls_seq), mpe))
    if world_size > 1:
        q = q.cuda()
        all_q = all_gather(q).float()
    else:
        all_q = q.float()
    q_mean = torch.mean(all_q).round(decimals=-1).item()
    q_std = torch.var(all_q).sqrt().round().item()
    print(
        f"Estimated tokens per sample {q_mean} with std {q_std} using {len(all_q)} samples and mpe {mpe}"
    )
    return q_mean


def special_sort(ls_idx: List[int], threshold: int = 3378606):
    print(f"applying special sort with threshold {threshold}")
    ls_idx = np.array(sorted(ls_idx))
    bool_sorted = ls_idx < threshold
    bool_shuffle = ls_idx >= threshold
    ls_idx_sorted = ls_idx[bool_sorted].tolist()
    ls_idx_shuffle = ls_idx[bool_shuffle].tolist()
    random.shuffle(ls_idx_shuffle)
    return ls_idx_sorted + ls_idx_shuffle


@torch.no_grad()
def dump_results(
    model,
    loader: DataLoader,
    device,
    writer,
    slice_id,
    epoch=1,
    ds_split="test",
):
    print(f"[{datetime.now()}] Start inference for epoch {epoch} ......")
    t_start = t_batch_start = datetime.now()
    report_gap = 10000

    indices = [0, 1, 2]
    i = i_prev = 0
    quotient = 1
    for j, test_data in enumerate(loader, 1):
        records = process_data(model, test_data, device)
        i += records.shape[0]
        if i // report_gap >= quotient:
            t_now = datetime.now()
            quotient = i // report_gap + 1
            time_elapse = (t_now - t_batch_start).total_seconds()
            tot_time_elapse = (t_now - t_start).total_seconds()
            print(
                f"[{slice_id}][epoch {epoch}][{ds_split}] Processed {i} samples, {round((i-i_prev)/time_elapse)} records/s, cost {time_elapse} s totally, {i/tot_time_elapse} records/s averagely"
            )
            t_batch_start = t_now
            i_prev = i
        writer.write(records, indices)  # write batch results


def process_data(model, test_data, device):
    input_ids = test_data["input_ids"].to(device)
    attention_mask = test_data["attention_mask"].to(device)
    cls_idx = test_data["cls_idx"].to(device) if "cls_idx" in test_data else None
    res = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        cls_idx=cls_idx,
    )
    # below for nodev2 type of task inferring
    logits = res.task_logits
    idx = test_data["idx"]
    assert isinstance(idx, np.ndarray), f"idx type: {type(idx)}"
    raw_node_idx = test_data["raw_node_idx"].to(device)
    ###### below similar to `metrics_utils.py::update`  ######
    # logits: [bz, seq, num_labels]
    # idx: [bz]
    # raw_node_idx: [bz, seq]
    assert len(logits.shape) in {2, 3}, f"logits shape: {logits.shape}"
    if len(logits.shape) == 3:
        y_pred = torch.argmax(logits, dim=-1)  # [bz, seq, vocab] -> [bz, seq]
    else:
        y_pred = logits
    assert len(y_pred.shape) == 2, f"y_pred shape: {y_pred.shape}"
    # remove nodes' dups
    raw_node_idx_new = raw_node_idx.reshape(-1)
    s_idx = torch.where(raw_node_idx_new != -100)[0].unique()
    raw_node_idx_new = raw_node_idx_new[s_idx]
    preds_new = y_pred.reshape(-1)[s_idx]
    # 1.1 reshape and expand idx
    seq = y_pred.shape[1]
    # [bz] -> [bz, 1] -> [bz, seq] -> [bz*seq]
    # idx_new = idx.reshape((-1, 1)).expand(-1, seq).reshape(-1)  # if idx is Tensor
    idx_new = np.tile(idx.reshape((-1, 1)), (1, seq)).reshape(-1)  # if idx is np.array
    idx_new = idx_new[s_idx.cpu().numpy()]
    # [bz*seq] & [bz*seq] & [bz*seq] -> [bz*seq, 3]
    records = list(
        zip(
            idx_new,
            raw_node_idx_new.cpu().numpy().astype(int),
            preds_new.cpu().numpy().astype(int),
        )
    )
    return records


def all_gather(q):
    """
    refer to: https://stackoverflow.com/a/71433508/4437068
    Gathers tensor arrays of different lengths, i.e., along dim=0, across multiple gpus

    Parameters
    ----------
        q : tensor array

    Returns
    -------
        all_q : concatenated gathered tensor arrays from all the gpus

    """
    device = q.device
    ws = dist.get_world_size()
    local_size = torch.tensor(q.shape[0], device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(all_sizes)

    size_diff = max_size.item() - local_size.item()
    if size_diff:
        new_shape = [size_diff] + list(q.shape[1:])
        padding = torch.zeros(new_shape, device=device, dtype=q.dtype)
        q = torch.cat([q, padding], dim=0)

    all_qs_padded = [torch.zeros_like(q) for _ in range(ws)]
    dist.all_gather(all_qs_padded, q)
    all_qs = []
    for q, size in zip(all_qs_padded, all_sizes):
        all_qs.append(q[:size])
    return torch.cat(all_qs)
