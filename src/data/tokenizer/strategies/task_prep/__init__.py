"""Task preparation strategies."""

from .base import TaskPreparationStrategy
from .pretrain import PretrainMLMStrategy, PretrainCoordStrategy
from .supervised import (
    GraphLevelStrategy,
    EdgeLevelStrategy,
    NodeLevelStrategy,
    NodeV2Strategy,
)

TASK_STRATEGY_MAP = {
    "pretrain": PretrainMLMStrategy,
    "pretrain-cl": PretrainMLMStrategy,
    "pretrain-mlm": PretrainMLMStrategy,
    "pretrain-smtp": PretrainCoordStrategy,
    "pretrain-coord-cl": PretrainCoordStrategy,
    "pretrain-coord": PretrainCoordStrategy,
    "pretrain-mlm-coord": PretrainCoordStrategy,
    "pretrain-ltp": PretrainMLMStrategy,
    "pretrain-euler": PretrainMLMStrategy,
    "graph": GraphLevelStrategy,
    "edge": EdgeLevelStrategy,
    "node": NodeLevelStrategy,
    "nodev2": NodeV2Strategy,
}


def get_task_strategy(task_type: str) -> TaskPreparationStrategy:
    """Factory function to get task preparation strategy."""
    if task_type not in TASK_STRATEGY_MAP:
        raise ValueError(f"Unknown task type: {task_type}")
    return TASK_STRATEGY_MAP[task_type]()


__all__ = [
    "TaskPreparationStrategy",
    "get_task_strategy",
    "PretrainMLMStrategy",
    "PretrainCoordStrategy",
    "GraphLevelStrategy",
    "EdgeLevelStrategy",
    "NodeLevelStrategy",
    "NodeV2Strategy",
]
