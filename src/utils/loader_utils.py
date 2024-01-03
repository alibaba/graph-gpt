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
        f"[world size {world_size}][rank {rank}][seed {seed}] raw-idx {len(sample_idx)}, new-idx {len(new_idx)}, new-idx top 10:\n{new_idx[:10]}"
    )
    return new_idx


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
