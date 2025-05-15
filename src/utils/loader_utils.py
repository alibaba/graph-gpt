import os
import random
import math
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import IterableDataset, Dataset, DataLoader


def set_up_shuffle_and_sampler(dataset, sampler):
    # IterableDataset is subclass of Dataset
    if isinstance(dataset, IterableDataset):
        shuffle = False
        sampler = None
        num_samples = len(dataset)
    elif isinstance(dataset, Dataset):
        shuffle = False if sampler is not None else True
        sampler = sampler
        num_samples = len(sampler) if sampler is not None else len(dataset)
    else:
        raise ValueError(f"dataset of type {type(dataset)} is NOT implemented!")
    return shuffle, sampler, num_samples


def distribute_sampler(sampler, world_size, rank):
    vec = np.array(sorted(sampler))
    idx = [i for i in range(len(sampler)) if i % world_size == rank]
    sampler = vec[idx].tolist()
    random.shuffle(sampler)
    return sampler


def distribute_sampler_with_rnd_seed(sample_idx, world_size, rank, seed):
    # deterministically shuffle based on epoch and seed
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(sample_idx), generator=g).tolist()
    # subsample
    total_size = (len(sample_idx) // world_size) * world_size
    indices = indices[rank:total_size:world_size]
    new_idx = sample_idx[indices].tolist()
    print(
        f"[distribute_sampler_with_rnd_seed][world size {world_size}][rank {rank}][seed {seed}] raw-idx {len(sample_idx)}, new-idx {len(new_idx)}, new-idx top 10:\n{new_idx[:10]}"
    )
    return new_idx


def obtain_deterministic_sampler(
    sample_idx: torch.Tensor, seed: int, cnt_samples: int = 10000
):
    # deterministically shuffle based on epoch and seed
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(sample_idx), generator=g)
    indices = indices[:cnt_samples]
    new_idx = sample_idx[indices].tolist()
    print(
        f"[obtain_deterministic_sampler][seed {seed}] raw-idx {len(sample_idx)}, new-idx {len(new_idx)}, new-idx top 10:\n{new_idx[:10]}"
    )
    return new_idx


def obtain_deterministic_sampler_by_ratio(
    sample_idx: torch.Tensor, seed: int, train: float = 0.8, valid: float = 0.1
):
    assert train + valid <= 1
    # deterministically shuffle based on epoch and seed
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(sample_idx), generator=g)
    cnt_train = int(len(sample_idx) * train)
    cnt_valid = int(len(sample_idx) * valid)

    indices_train = indices[:cnt_train]
    indices_valid = indices[cnt_train : cnt_train + cnt_valid]
    indices_test = indices[cnt_train + cnt_valid :]

    idx_train = sample_idx[indices_train].tolist()
    idx_valid = sample_idx[indices_valid].tolist()
    idx_test = sample_idx[indices_test].tolist()

    print(
        f"[obtain_deterministic_sampler][seed {seed}] raw-idx {len(sample_idx)}, idx_train {len(idx_train)}, idx_valid {len(idx_valid)}, idx_test {len(idx_test)}\n"
        f"TOP 10 sample_idx: {sample_idx[:10]}"
    )
    return idx_train, idx_valid, idx_test


# Define a `worker_init_fn` that configures each iterable-dataset copy differently
def worker_init_fn(worker_id):
    # refer to: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(
        math.ceil((overall_end - overall_start) / float(worker_info.num_workers))
    )
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


def worker_init_fn_seed(worker_id):
    # seeding for each worker: https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading
    seed = torch.utils.data.get_worker_info().seed
    random.seed(seed)
    np_seed = seed % 2**32
    np.random.seed(np_seed)
    print(
        f"[DataLoader Worker {worker_id}] seed `random` & `np.random` with {seed} & {np_seed}!"
    )


def load_from_ckp(
    misc_utils,
    pretrain_cpt,
    output_dir,
    model,
    config,
    skip_keys=True,
    strict=False,
):
    if (len(pretrain_cpt) > 0) and (pretrain_cpt != output_dir):
        ckp, prev_epoch = misc_utils.get_latest_ckp(pretrain_cpt)
        model = load_from_ckp_with_try(model, ckp, config, skip_keys, strict)
    return model


def load_from_ckp_with_try(
    model,
    ckp,
    config,
    skip_keys=True,
    strict=False,
    use_ema=False,
):
    print(f"Loading pretrained weights from ckp {ckp}")
    try:
        if use_ema:
            fn_model = os.path.join(ckp, "../model_ema_best.pt")
            # if not os.path.isfile(fn_model):
        else:
            fn_model = os.path.join(ckp, "model.pt")
        print(f"[{datetime.now()}] loading ckp using torch API from:\n{fn_model} ...")
        stat_dict = torch.load(fn_model)
        stat_dict = {
            (k[7:] if k.startswith("module.") else k): v for k, v in stat_dict.items()
        }
        print(f"[{datetime.now()}] ckp loaded:\n{fn_model}")
    except Exception as inst:
        # print(type(inst))
        # print(inst.args)
        print(inst)
        from deepspeed.utils.zero_to_fp32 import (
            get_fp32_state_dict_from_zero_checkpoint,
        )

        stat_dict = get_fp32_state_dict_from_zero_checkpoint(ckp)
        print(
            f"[{datetime.now()}] load ckp using DeepSpeed API `get_fp32_state_dict_from_zero_checkpoint`"
        )

    for key in list(stat_dict.keys()):
        if ("score" in key) and skip_keys:
            stat_dict.pop(key)
            print(f"pop key {key} in stat_dict!")
    missing_keys, unexpected_keys = model.load_state_dict(stat_dict, strict=strict)
    print(
        f"[{datetime.now()}] init model params using pytorch `load_state_dict`\n"
        f"missing keys: {missing_keys}\n"
        f"unexpected_keys: {unexpected_keys}\n"
        f"After loading weights from ckp:\n{model.config}\nnum_labels: {model.num_labels if hasattr(model, 'num_labels') else None}\nmodel-type: {model.dtype}\n\n{model}"
    )
    return model


def get_train_valid_test_sampler(
    train_dataset,
    valid_dataset,
    test_dataset,
    batch_size,
    k_samplers,
    world_size,
    rank,
    num_workers,
):
    rnd_seed = torch.random.initial_seed() - rank
    random.seed(rnd_seed)
    if not isinstance(train_dataset, IterableDataset):
        train_sampler = train_dataset.sampler
        random.shuffle(train_sampler)
        train_shuffle, train_sampler, train_cnt = set_up_shuffle_and_sampler(
            train_dataset, train_sampler
        )
        train_sampler_for_eval = random.sample(
            train_sampler, min(len(train_sampler), k_samplers)
        )

        valid_sampler = (
            valid_dataset.sampler
            if hasattr(valid_dataset, "sampler")
            else list(range(len(valid_dataset)))
        )
        random.shuffle(valid_sampler)
        valid_shuffle, valid_sampler, valid_cnt = set_up_shuffle_and_sampler(
            valid_dataset, valid_sampler
        )
        valid_sampler = distribute_sampler(valid_sampler, world_size, rank)
        valid_sampler_for_eval = random.sample(
            valid_sampler, min(len(valid_sampler), k_samplers)
        )

        test_sampler = (
            test_dataset.sampler
            if hasattr(test_dataset, "sampler")
            else list(range(len(test_dataset)))
        )
        random.shuffle(test_sampler)
        test_shuffle, test_sampler, test_cnt = set_up_shuffle_and_sampler(
            test_dataset, test_sampler
        )
        test_sampler = distribute_sampler(test_sampler, world_size, rank)
        test_sampler_for_eval = random.sample(
            test_sampler, min(len(test_sampler), k_samplers)
        )

        steps_per_epoch = (len(train_sampler) // world_size) // batch_size
    else:
        train_cnt = len(train_dataset) * world_size
        train_sampler = None
        train_sampler_for_eval = None
        train_shuffle = False

        valid_cnt = len(valid_dataset) * world_size
        valid_sampler = None
        valid_sampler_for_eval = None

        test_cnt = len(test_dataset) * world_size
        test_sampler = None
        test_sampler_for_eval = None

        steps_per_epoch = (
            (len(train_dataset) // num_workers) // batch_size
        ) * num_workers
    return (
        train_cnt,
        train_sampler,
        train_sampler_for_eval,
        train_shuffle,
        valid_cnt,
        valid_sampler,
        valid_sampler_for_eval,
        test_cnt,
        test_sampler,
        test_sampler_for_eval,
        steps_per_epoch,
    )


def get_eval_loader(
    train_dataset,
    valid_dataset,
    test_dataset,
    batch_size_eval,
    train_sampler_for_eval,
    valid_sampler,
    test_sampler,
    num_workers_eval,
    collator_fn,
):
    train_loader_for_eval = (
        DataLoader(
            dataset=train_dataset,
            batch_size=batch_size_eval,
            sampler=train_sampler_for_eval,
            num_workers=num_workers_eval,
            collate_fn=collator_fn,
            worker_init_fn=worker_init_fn_seed,
            pin_memory=True,
        )
        if not isinstance(train_dataset, IterableDataset)
        else None
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size_eval,
        sampler=valid_sampler,
        num_workers=num_workers_eval,
        collate_fn=collator_fn,
        worker_init_fn=worker_init_fn_seed,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_eval,
        sampler=test_sampler,
        num_workers=num_workers_eval,
        collate_fn=collator_fn,
        worker_init_fn=worker_init_fn_seed,
        pin_memory=True,
    )
    return train_loader_for_eval, valid_loader, test_loader


def init_loader_for_odps_table_ds(
    epoch,
    epoch_start,
    j,
    steps_per_epoch,
    batch_size,
    OdpsTableIterableDataset,
    tables,
    edge_dim,
    node_dim,
    y_dim,
    train_shuffle,
    train_sampler,
    num_workers,
    collator_fn,
):
    if epoch == epoch_start:
        skipped_steps = j % steps_per_epoch
        skipped_samples = skipped_steps * batch_size  # per gpu
        i_local = skipped_steps
    else:
        skipped_samples = 0
        i_local = 0
    print(
        f"reset dataset and train_loader for epoch {epoch} with skipped samples per gpu: {skipped_samples}"
    )
    train_dataset = OdpsTableIterableDataset(
        table_path=tables,
        slice_id=int(os.environ.get("RANK", 0)),
        slice_count=int(os.environ.get("WORLD_SIZE", 1)),
        skipped_samples=skipped_samples,
        permute_nodes=True,
        epoch=epoch,
        edge_dim=edge_dim,
        node_dim=node_dim,
        y_dim=y_dim,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collator_fn,
        worker_init_fn=worker_init_fn_seed,
        pin_memory=True,
        drop_last=False,  # set False for odps dataset
        prefetch_factor=4,
    )
    return train_loader, i_local
