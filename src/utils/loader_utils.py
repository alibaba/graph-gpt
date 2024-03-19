import random
import math
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset


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
        f"[obtain_deterministic_sampler][seed {seed}] raw-idx {len(sample_idx)}, idx_train {len(idx_train)}, idx_valid {len(idx_valid)}, idx_test {len(idx_test)}"
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
