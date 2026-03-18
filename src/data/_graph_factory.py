"""Generic graph-dataset factory driven by ``DatasetSpec`` declarations.

Every spec is a pure data record that captures *what* differs between
graph-level dataset readers (constructor class, split logic, label
transform, etc.).  The single ``read_graph_dataset`` function interprets
a spec and returns the same ``(train_ds, [valid_ds, test_ds,] raw_dataset)``
tuple that the old hand-written readers produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

import torch

from .dataset_map import GraphsMapDataset


@dataclass
class DatasetSpec:
    """Declarative description of a graph-level dataset source."""

    name: str
    dataset_cls: Any  # class used to build the dataset
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    also_molecule: bool = False

    # --- split configuration ---
    split_method: str = "get_idx_split"  # "get_idx_split" | "hardcoded" | "random"
    hardcoded_splits: Optional[Dict[str, slice]] = None
    random_split_seed: Optional[int] = None
    random_train_ratio: float = 0.8
    valid_slice: Optional[slice] = None
    test_slice: Optional[slice] = None
    test_from_valid: bool = False

    # --- pretrain mode ---
    pretrain_only: bool = False
    pretrain_permute_nodes: bool = False
    pretrain_sample_idx: Union[str, Callable] = "all"  # "all" | "train_split" | callable

    # --- finetune mode ---
    ft_permute_nodes: bool = False

    # --- hooks ---
    label_transform: Optional[Callable] = None  # fn(dataset) -> None
    post_load_hook: Optional[Callable] = None  # fn(dataset) -> None


def read_graph_dataset(spec: DatasetSpec, data_cfg, *, with_prob: bool = False, **kwargs):
    """Generic reader for any dataset described by *spec*."""
    data_dir = data_cfg.data_dir
    return_valid_test = data_cfg.return_valid_test

    if spec.pretrain_only:
        assert not return_valid_test, f"{spec.name} is pretrain-only"

    print(f"\nLoading dataset {spec.name} ...")
    dataset = spec.dataset_cls(root=data_dir, **spec.dataset_kwargs)
    print(f"\ndataset._data -> {dataset._data}")

    if spec.label_transform is not None:
        spec.label_transform(dataset)
    if spec.post_load_hook is not None:
        spec.post_load_hook(dataset)

    if return_valid_test:
        train_idx, valid_idx, test_idx = _resolve_splits(dataset, spec)
        train_dataset = GraphsMapDataset(
            dataset, None,
            sample_idx=train_idx,
            permute_nodes=spec.ft_permute_nodes,
            provide_sampler=True,
        )
        valid_dataset = GraphsMapDataset(
            dataset, None,
            sample_idx=valid_idx,
            permute_nodes=spec.ft_permute_nodes,
            provide_sampler=True,
        )
        test_dataset = GraphsMapDataset(
            dataset, None,
            sample_idx=test_idx,
            permute_nodes=spec.ft_permute_nodes,
            provide_sampler=True,
        )
        print(
            f"Split dataset based on given train/valid/test index!\n"
            f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return train_dataset, valid_dataset, test_dataset, dataset
    else:
        sample_idx = _resolve_pretrain_idx(dataset, spec)
        train_dataset = GraphsMapDataset(
            dataset, None,
            sample_idx=sample_idx,
            permute_nodes=spec.pretrain_permute_nodes,
            provide_sampler=True,
            with_prob=with_prob,
        )
        return train_dataset, dataset


def _resolve_splits(dataset, spec: DatasetSpec):
    """Return (train_idx, valid_idx, test_idx) tensors."""
    if spec.split_method == "get_idx_split":
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]
        valid_idx = split_idx["valid"]
        if spec.test_from_valid:
            test_idx = split_idx["valid"].clone()
        else:
            test_idx = split_idx.get("test", split_idx["valid"])
    elif spec.split_method == "hardcoded":
        indices = torch.arange(len(dataset))
        h = spec.hardcoded_splits
        train_idx = indices[h["train"]]
        valid_idx = indices[h["valid"]]
        test_idx = indices[h["test"]]
    elif spec.split_method == "random":
        g = torch.Generator()
        if spec.random_split_seed is not None:
            g.manual_seed(spec.random_split_seed)
        indices = torch.randperm(len(dataset), generator=g)
        train_max = int(len(dataset) * spec.random_train_ratio)
        train_idx = indices[:train_max]
        valid_idx = indices[train_max:]
        test_idx = indices[-8:]
    else:
        raise ValueError(f"Unknown split_method: {spec.split_method}")

    if spec.valid_slice is not None:
        valid_idx = valid_idx[spec.valid_slice]
    if spec.test_slice is not None:
        test_idx = test_idx[spec.test_slice]

    return train_idx, valid_idx, test_idx


def _resolve_pretrain_idx(dataset, spec: DatasetSpec):
    """Return the sample-index tensor for pretrain (no valid/test)."""
    if callable(spec.pretrain_sample_idx):
        return spec.pretrain_sample_idx(dataset)
    if spec.pretrain_sample_idx == "train_split":
        return dataset.get_idx_split()["train"]
    # default: "all"
    return torch.arange(len(dataset))


def register_specs(specs, dataset_registry, molecule_registry):
    """Register a list of DatasetSpec instances into the given registries."""
    for spec in specs:
        def _make_reader(s):
            def reader(data_cfg, **kwargs):
                return read_graph_dataset(s, data_cfg, **kwargs)
            return reader
        dataset_registry(spec.name)(_make_reader(spec))
        if spec.also_molecule:
            molecule_registry(spec.name)(_make_reader(spec))
