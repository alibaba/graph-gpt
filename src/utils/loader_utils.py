import os
import random
import math
import numpy as np
import copy
import torch
from datetime import datetime
from torch.utils.data import IterableDataset, Dataset, DataLoader
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Any

from . import inspect_tokenization_results
from ..conf import TrainingConfig, TrainingStats, Config, DataConfig


@dataclass
class SamplerConfig:
    sampler: Optional[List[int]] = None
    # pick a subset from sampler to run evaluation
    sampler_for_eval: Optional[List[int]] = None
    cnt: Optional[int] = None
    shuffle: bool = False

    ds: Optional[Dataset] = None


@dataclass(frozen=True)  # use frozen=True for immutable，like NamedTuple
class PTSamplerConfig:
    train_sampler: Optional[List[int]]
    train_cnt: int
    train_shuffle: bool = False

    sampler_for_eval: Optional[List[int]] = None

    test_dataset: Optional[Dataset] = None
    sampler_for_test: Optional[List[int]] = None


@dataclass
class FTSamplerConfig:  # finetuning sampler
    train: SamplerConfig = field(default_factory=SamplerConfig)
    valid: SamplerConfig = field(default_factory=SamplerConfig)
    test: SamplerConfig = field(default_factory=SamplerConfig)

    def enlarge_valid_test_samples(self, eval_only: bool, samp_rate: int = 10):
        if eval_only:
            self.valid.sampler = list(self.valid.sampler * samp_rate)
            self.test.sampler = list(self.test.sampler * samp_rate)
            self.valid.cnt = self.valid.cnt * samp_rate
            self.test.cnt = self.test.cnt * samp_rate
            print(f"Enlarge valid/test by {samp_rate} times for ensemble ...")


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


def set_train_valid_test_sampler(
    ft_sampler: FTSamplerConfig, train_cfg: TrainingConfig
):
    train_dataset = ft_sampler.train.ds
    valid_dataset = ft_sampler.valid.ds
    test_dataset = ft_sampler.test.ds
    batch_size = train_cfg.batch_size
    k_samplers = train_cfg.ft_eval.k_samplers
    world_size = train_cfg.distributed.world_size
    rank = train_cfg.distributed.rank
    num_workers = train_cfg.num_workers

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
    ft_sampler.train.cnt = train_cnt
    ft_sampler.train.shuffle = train_shuffle
    ft_sampler.train.sampler = train_sampler
    ft_sampler.train.sampler_for_eval = train_sampler_for_eval

    ft_sampler.valid.cnt = valid_cnt
    ft_sampler.valid.sampler = valid_sampler
    ft_sampler.valid.sampler_for_eval = valid_sampler_for_eval

    ft_sampler.test.cnt = test_cnt
    ft_sampler.test.sampler = test_sampler
    ft_sampler.test.sampler_for_eval = test_sampler_for_eval
    return steps_per_epoch


def get_cl_sampler(sampler: list, dup: bool = False):
    if dup:
        tmp_len = len(sampler)
    else:
        tmp_len = len(sampler) // 2
    sampler = [x for ele in zip(sampler[:tmp_len], sampler[:tmp_len]) for x in ele]
    print(f"top 10 sampler for CL loss:\n{sampler[:10]}")
    return sampler


def get_pt_train_valid_test_sampler(
    train_dataset,
    training: TrainingConfig,
    task_type: str,
    data_cfg: DataConfig,
    read_dataset,
    raw_dataset=None,
) -> PTSamplerConfig:
    # sampler for pre-training
    if not isinstance(train_dataset, IterableDataset):
        train_sampler = train_dataset.sampler
        random.shuffle(train_sampler)
        train_shuffle, train_sampler, train_cnt = set_up_shuffle_and_sampler(
            train_dataset, train_sampler
        )
    else:
        train_cnt = len(train_dataset) * training.distributed.world_size
        train_sampler = None
        train_shuffle = False

    sampler_for_eval = []
    if training.do_valid:
        # v1. split here
        (
            train_sampler,
            sampler_eval,
            _,
        ) = obtain_deterministic_sampler_by_ratio(
            train_dataset.sample_idx,
            seed=42,
            train=1 - training.valid_percent,
            valid=training.valid_percent,
        )
        # v2. split from idx
        # train_sampler = raw_dataset.get_idx_split()["train"].tolist()
        # sampler_for_eval = raw_dataset.get_idx_split()["valid"].tolist()
        # sampler_for_eval = random.sample(
        #     raw_dataset.get_idx_split()["valid"].tolist(), 10000
        # )
        print(
            f"[{datetime.now()}] train samples reduced from {len(train_dataset.sampler)} -> {len(train_sampler)}"
        )
        train_dataset.sampler = train_sampler
        sampler_for_eval = distribute_sampler(
            sampler_eval, training.distributed.world_size, training.distributed.rank
        )
        if task_type.endswith("-cl"):
            sampler_for_eval = get_cl_sampler(sampler_for_eval, True)
        print(
            f"[{datetime.now()}] sampler_for_eval: {len(sampler_eval)}, top 10: {sampler_for_eval[:10]}"
        )
    test_dataset = None
    sampler_for_test = None
    if training.do_test:
        # test_dataset, _ = read_dataset(
        #     name="reddit_threads",
        #     # for local data file reading
        #     data_cfg=dataclasses.replace(data_cfg, data_dir=os.path.join(data_dir, "../TUDataset")),
        #     train_cfg = training,
        # )
        # sampler_for_test = loader_utils.obtain_deterministic_sampler(
        #     test_dataset.sample_idx, seed=42, cnt_samples=20000
        # )
        print(f"Using ZINC dataset for validating loss.")
        test_dataset, _ = read_dataset(
            name="ZINC", data_cfg=data_cfg, train_cfg=training
        )
        sampler_for_test = distribute_sampler_with_rnd_seed(
            test_dataset.sample_idx,
            training.distributed.world_size,
            training.distributed.rank,
            seed=42,
        )[:20000]
        if task_type.endswith("-cl"):
            sampler_for_test = get_cl_sampler(sampler_for_test, True)
    pt_sampler = PTSamplerConfig(
        train_sampler=train_sampler,
        train_cnt=train_cnt,
        train_shuffle=train_shuffle,
        sampler_for_eval=sampler_for_eval,
        test_dataset=test_dataset,
        sampler_for_test=sampler_for_test,
    )
    return pt_sampler


def reset_pt_train_sampler(
    reset_samples_per_epoch: bool,
    task_type: str,
    train_dataset: Dataset,
    training: TrainingConfig,
    pt_sampler: PTSamplerConfig,
) -> PTSamplerConfig:
    sched_cfg = training.schedule
    rank = training.distributed.rank
    if (not reset_samples_per_epoch) and (
        not isinstance(train_dataset, IterableDataset)
    ):
        train_sampler_new = []
        for epoch in range(sched_cfg.epochs):
            train_dataset.reset_samples(epoch, rank)
            # random.shuffle(train_sampler)
            train_sampler_new.extend(train_dataset.sampler)
        random.shuffle(train_sampler_new)
        print(
            f"train_sampler for {sched_cfg.epochs} epochs increase: {len(pt_sampler.train_sampler)} -> {len(train_sampler_new)}\n"
        )
        pt_sampler = dataclasses.replace(pt_sampler, train_sampler=train_sampler_new)
        if task_type.endswith("-cl"):
            pt_sampler = dataclasses.replace(
                pt_sampler,
                train_sampler=get_cl_sampler(pt_sampler.train_sampler, False),
            )
        sched_cfg.epochs = 1
        print(f"reset actual epochs to {sched_cfg.epochs}")
    return pt_sampler


def get_eval_loader(
    ft_sampler: FTSamplerConfig, train_cfg: TrainingConfig, collator_fn
):
    train_loader_for_eval = (
        DataLoader(
            dataset=ft_sampler.train.ds,
            batch_size=train_cfg.batch_size_eval,
            sampler=ft_sampler.train.sampler_for_eval,
            num_workers=train_cfg.num_workers_eval,
            collate_fn=collator_fn,
            worker_init_fn=worker_init_fn_seed,
            pin_memory=True,
        )
        if not isinstance(ft_sampler.train.ds, IterableDataset)
        else None
    )
    valid_loader = DataLoader(
        dataset=ft_sampler.valid.ds,
        batch_size=train_cfg.batch_size_eval,
        sampler=ft_sampler.valid.sampler,
        num_workers=train_cfg.num_workers_eval,
        collate_fn=collator_fn,
        worker_init_fn=worker_init_fn_seed,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=ft_sampler.test.ds,
        batch_size=train_cfg.batch_size_eval,
        sampler=ft_sampler.test.sampler,
        num_workers=train_cfg.num_workers_eval,
        collate_fn=collator_fn,
        worker_init_fn=worker_init_fn_seed,
        pin_memory=True,
    )
    return train_loader_for_eval, valid_loader, test_loader


def init_odps_ds_stats(
    training: TrainingConfig,
    train_stats: TrainingStats,
    pt_sampler: PTSamplerConfig,
):
    if train_stats.odps_stats.is_odps_table_ds:
        world_size = training.distributed.world_size
        batch_size = training.batch_size
        steps_per_epoch = pt_sampler.train_cnt // (world_size * batch_size)
        print(
            f"\nsteps_per_epoch: {steps_per_epoch} = {pt_sampler.train_cnt} // ({world_size} * {batch_size})"
        )
        epoch_start = train_stats.j // steps_per_epoch
        print(
            f"pre-train using odps table, set epoch_start={epoch_start} in case of resuming pre-training"
        )
        train_stats.odps_stats.steps_per_epoch = steps_per_epoch
    else:
        epoch_start = 0
    train_stats.epoch_start = epoch_start


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


def initialize_train_loader_at_epoch_start(
    train_dataset: Dataset,
    train_cfg: TrainingConfig,
    train_stats: TrainingStats,
    pt_sampler: PTSamplerConfig,
    collator_fn,
    OdpsTableIterableDataset,
):
    train_sampler = pt_sampler.train_sampler
    train_loader = None
    if (
        not isinstance(train_dataset, IterableDataset)
    ) and train_stats.reset_samples_per_epoch:
        print(f"Re-initialize train-loader with shuffled sampler and reset dataset!")
        train_dataset.reset_samples(train_stats.epoch, train_cfg.distributed.rank)
        if not train_cfg.do_valid:
            train_sampler = train_dataset.sampler
        random.shuffle(train_sampler)
        print(f"train_sampler: {len(train_sampler)}")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=pt_sampler.train_shuffle,
            sampler=train_sampler,
            num_workers=train_cfg.num_workers,
            collate_fn=collator_fn,
            worker_init_fn=worker_init_fn_seed,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
        )
    i_local = 0
    if train_stats.odps_stats.is_odps_table_ds:
        train_loader, i_local = init_loader_for_odps_table_ds(
            epoch=train_stats.epoch,
            epoch_start=train_stats.epoch_start,
            j=train_stats.j,
            steps_per_epoch=train_stats.odps_stats.steps_per_epoch,
            batch_size=train_cfg.batch_size,
            OdpsTableIterableDataset=OdpsTableIterableDataset,
            tables=train_stats.odps_stats.tables,
            edge_dim=train_dataset.edge_dim,
            node_dim=train_dataset.node_dim,
            y_dim=train_dataset.y_dim,
            train_shuffle=pt_sampler.train_shuffle,
            train_sampler=train_sampler,
            num_workers=train_cfg.num_workers,
            collator_fn=collator_fn,
        )

    train_stats.i_local = i_local
    return train_loader


def initialize_ft_train_loader_at_epoch_start(
    train_dataset: Dataset,
    train_cfg: TrainingConfig,
    train_stats: TrainingStats,
    ft_sampler: FTSamplerConfig,
    collator_fn,
):
    print(f"Re-initialize train-loader!")
    train_sampler = ft_sampler.train.sampler
    if not isinstance(train_dataset, IterableDataset):
        print(f"Shuffle sampler and reset dataset!")
        train_dataset.reset_samples(
            train_stats.epoch, train_cfg.finetune.seed
        ) if hasattr(train_dataset, "reset_samples") else None
        train_sampler = distribute_sampler_with_rnd_seed(
            torch.tensor(train_dataset.sampler),
            train_cfg.distributed.world_size,
            train_cfg.distributed.rank,
            seed=train_cfg.finetune.seed + train_stats.epoch,
        )
    # train_sampler = misc_utils.special_sort(train_sampler)
    collator_fn.is_training = True
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=train_cfg.num_workers,
        collate_fn=collator_fn,
        worker_init_fn=worker_init_fn_seed,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
    )
    return train_loader


def get_valid_test_loader(train_dataset, collator_fn, sampler_for_eval):
    batch_size_eval = 128  # small to avoid OOM when evaluating
    num_workers_eval = 8
    loader_for_eval = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_eval,
        sampler=sampler_for_eval,
        num_workers=num_workers_eval,
        collate_fn=collator_fn,
        worker_init_fn=worker_init_fn_seed,
        pin_memory=False,
    )
    return loader_for_eval


def initialize_pt_valid_loader(
    train_dataset: Dataset,
    cfg: Config,
    pt_sampler: PTSamplerConfig,
    tokenizer_config: dict[str, Any],
    tokenizer_cls,  # src.data.tokenizer.GSTTokenizer|StackedGSTTokenizer
    collator_cls,  # src.data.collator.DataCollatorForGST
):
    token_cfg = cfg.tokenization
    model_cfg = cfg.model
    train_cfg = cfg.training

    valid_loader = None
    valgen_loader = None
    if train_cfg.do_valid:
        valid_tokenizer_config = copy.deepcopy(tokenizer_config)
        # valid_tokenizer_config["structure"]["nx"]["func"] = [
        #     {"name": "shortest_path_length", "valid": 1}
        # ]
        # gtokenizer_valid = tokenizer_cls(valid_tokenizer_config, add_eos=False)
        # if valid_tokenizer_config["structure"]["nx"]["enable"]:
        #     gtokenizer_valid.attr_mask_ratio = 1  # mask all attr
        # inspect_tokenization_results(train_dataset, gtokenizer_valid)
        # gtokenizer_valid = gtokenizer
        gtokenizer_valid = tokenizer_cls(
            valid_tokenizer_config,
            add_eos=token_cfg.add_eos,
            stack_method=model_cfg.graph_input.stack_method,
            train_cfg=train_cfg,
        )
        valid_loader = get_valid_test_loader(
            train_dataset,
            collator_cls(
                tokenizer=gtokenizer_valid,
                max_length=train_cfg.max_length,
                pad_to_multiple_of=train_cfg.pad_to_multiple_of,
                return_tensors="pt",
            ),
            pt_sampler.sampler_for_eval,
        )
        if train_cfg.do_generation:
            valgen_loader = get_valid_test_loader(
                train_dataset,
                collator_cls(
                    tokenizer=gtokenizer_valid,
                    max_length=train_cfg.max_length,
                    pad_to_multiple_of=train_cfg.pad_to_multiple_of,
                    return_tensors="pt",
                ),
                pt_sampler.sampler_for_eval[: train_cfg.pretrain_mlm.num_gen_samples],
            )
    return valid_loader, valgen_loader


def initialize_pt_test_loader(
    gtokenizer,  # src.data.tokenizer.GSTTokenizer|StackedGSTTokenizer
    cfg: Config,
    pt_sampler: PTSamplerConfig,
    tokenizer_config: dict,
    tokenizer_cls,  # src.data.tokenizer.GSTTokenizer|StackedGSTTokenizer
    collator_cls,  # src.data.collator.DataCollatorForGST
):
    model_config = cfg.model
    training = cfg.training

    test_loader = None
    if training.do_test:
        # test_tokenizer_config = copy.deepcopy(tokenizer_config)
        # test_tokenizer_config["semantics"]["node"]["discrete"] = None
        # test_tokenizer_config["semantics"]["edge"]["discrete"] = None
        # test_tokenizer_config["structure"]["nx"]["func"] = [
        #     {"name": "shortest_path_length", "valid": 1}
        # ]
        # gtokenizer_test = tokenizer_cls(test_tokenizer_config, add_eos=False)
        # gtokenizer_test = tokenizer_cls(
        #     tokenizer_config, add_eos=add_eos, stack_method=stack_method, train_cfg=training,
        # )
        gtokenizer_test = gtokenizer
        # gtokenizer_test.attr_mask_ratio = 0  # no need mask
        inspect_tokenization_results(pt_sampler.test_dataset, gtokenizer_test)
        test_loader = get_valid_test_loader(
            pt_sampler.test_dataset,
            collator_cls(
                tokenizer=gtokenizer_test,
                max_length=model_config.max_position_embeddings,
                pad_to_multiple_of=training.pad_to_multiple_of,
                return_tensors="pt",
            ),
            pt_sampler.sampler_for_test,
        )
    return test_loader
