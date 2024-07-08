import os
import time
import shutil
import random
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset


def delete_old_ckp(dir_ckp):
    if os.path.exists(dir_ckp):
        shutil.rmtree(dir_ckp, ignore_errors=True)


def get_latest_ckp(pretrain_cpt):
    dirs_scan = [f.path for f in os.scandir(pretrain_cpt) if f.is_dir()]
    dirs_scan = [dir_.split("/")[-1] for dir_ in dirs_scan]
    dirs_scan = [dir_.split("_")[-1] for dir_ in dirs_scan]
    dirs_scan = [int(dir_) for dir_ in dirs_scan if dir_.isnumeric()]
    ckp = os.path.join(pretrain_cpt, f"epoch_{max(dirs_scan)}")
    return ckp, max(dirs_scan)


def convert_dict_to_df(dict_):
    dict_ = {key: val.float().detach().cpu().numpy() for key, val in dict_.items()}
    return pd.DataFrame(dict_)


def save_pred_results(dict_, model_dir, name):
    df = convert_dict_to_df(dict_)
    fn = os.path.join(model_dir, f"{name}_results.csv")
    df.to_csv(fn, index=False)
    print(f"{name} results saved in {fn}!")


def save_ckp(output_dir, model, epoch, use_ddp):
    rank = int(os.environ.get("RANK", 0))
    if use_ddp:
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
            save_all(output_dir, model, epoch, save_model=True)


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
):
    model_dir = os.path.join(output_dir, f"epoch_{epoch}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    # a). save model ckp
    if save_model:
        model.save_pretrained(model_dir)
        print(f"Model saved in {model_dir} using HF API.")
    # b). delete old ckp
    old_model_dir = os.path.join(output_dir, f"epoch_{epoch - 5*world_size}")
    delete_old_ckp(old_model_dir)
    # c). save train/eval loss/metrics logs
    if ls_log:
        fn_log = f"{output_dir}/log.csv"
        with open(fn_log, "w+") as fp:
            fp.writelines(ls_log)
        print(f"log saved in {fn_log}!")
    # d). save eval/test metrics
    if ls_result:
        fn_result = f"{output_dir}/result.csv"
        with open(fn_result, "w+") as fp:
            fp.writelines(ls_result)
        print(f"log saved in {fn_result}!")
    # e). save fine-tune training loss
    if ls_loss:
        fn_loss = f"{output_dir}/loss.csv"
        with open(fn_loss, "w+") as fp:
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


def estimate_tokens_per_sample(
    gtokenizer, dataset, sampler, mpe, world_size, tot_samples: int = 10000
):
    num_samples = tot_samples // world_size
    if not isinstance(dataset, IterableDataset):
        num_samples = min(num_samples, len(sampler))
        sampler = random.sample(sampler, num_samples)
        ls_seq = [
            len(gtokenizer.tokenize(dataset[idx][-1])[0]) for idx in tqdm(sampler)
        ]
    else:
        ls_seq = []
        print(f"Estimating tokens_per_sample ...")
        for i, ele in enumerate(iter(dataset)):
            if i >= num_samples:
                break
            ls_seq.append(len(gtokenizer.tokenize(ele[-1])[0]))
    if world_size > 1:
        seq_sum = torch.tensor(np.sum(np.minimum(np.array(ls_seq), mpe))).cuda()
        dist.all_reduce(seq_sum)
        avg_len = round(seq_sum.item() / (world_size * num_samples), -1)
    else:
        avg_len = np.mean(np.minimum(np.array(ls_seq), mpe)).round()
    print(
        f"Estimated tokens per sample {avg_len} with {world_size*num_samples} samples and mpe {mpe}"
    )
    return avg_len


def special_sort(ls_idx: List[int], threshold: int = 3378606):
    print(f"applying special sort with threshold {threshold}")
    ls_idx = np.array(sorted(ls_idx))
    bool_sorted = ls_idx < threshold
    bool_shuffle = ls_idx >= threshold
    ls_idx_sorted = ls_idx[bool_sorted].tolist()
    ls_idx_shuffle = ls_idx[bool_shuffle].tolist()
    random.shuffle(ls_idx_shuffle)
    return ls_idx_sorted + ls_idx_shuffle
