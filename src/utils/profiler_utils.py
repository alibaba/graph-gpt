"""PyTorch Profiler utilities for analyzing GPU performance bottlenecks.

This module provides a context manager and utilities for profiling training loops
to identify performance bottlenecks like:
- GPU duty cycle and SM utilization
- CUDA kernel execution time
- Memory operations and data transfer overhead
- CPU-GPU synchronization points

Usage:
    from src.utils.profiler_utils import TrainingProfiler

    profiler = TrainingProfiler(
        output_dir="./profiler_output",
        wait_steps=5,
        warmup_steps=5,
        active_steps=10,
        repeat=2,
    )

    for step, data in enumerate(dataloader):
        with profiler.step(step):
            # training step
            ...

    profiler.export_summary()
"""

import os
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
    record_function,
)


@dataclass
class ProfilerConfig:
    """Configuration for PyTorch Profiler.

    Attributes:
        enabled: Whether to enable profiling (default: False)
        output_dir: Directory to save profiler outputs
        wait_steps: Number of steps to wait before starting profiling
        warmup_steps: Number of warmup steps (profiler is on but results discarded)
        active_steps: Number of steps to actively profile
        repeat: Number of profiling cycles to repeat (0 = profile once then disable)
        record_shapes: Whether to record tensor shapes
        profile_memory: Whether to profile memory usage
        with_stack: Whether to record source code stack traces
        with_flops: Whether to estimate FLOPs for each operator
        with_modules: Whether to record module hierarchy
        export_chrome_trace: Whether to export Chrome trace format
        export_stacks: Whether to export stack traces
        tensorboard: Whether to export TensorBoard logs
    """

    enabled: bool = False
    output_dir: str = "./profiler_output"
    wait_steps: int = 5
    warmup_steps: int = 5
    active_steps: int = 10
    repeat: int = 1
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True
    with_flops: bool = True
    with_modules: bool = True
    export_chrome_trace: bool = True
    export_stacks: bool = False
    tensorboard: bool = True


class TrainingProfiler:
    """Context manager for profiling PyTorch training loops.

    This profiler wraps torch.profiler.profile with sensible defaults for
    analyzing GPU performance in training scenarios. It supports:
    - Automatic step scheduling with wait/warmup/active phases
    - TensorBoard integration for visualization
    - Chrome trace export for detailed analysis
    - Summary statistics for quick performance overview

    Example:
        profiler = TrainingProfiler(output_dir="./profiles", active_steps=20)

        for step, batch in enumerate(train_loader):
            with profiler.step(step):
                model(batch)
                loss.backward()
                optimizer.step()

        profiler.export_summary()
    """

    def __init__(
        self,
        output_dir: str = "./profiler_output",
        wait_steps: int = 5,
        warmup_steps: int = 5,
        active_steps: int = 10,
        repeat: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
        with_flops: bool = True,
        with_modules: bool = True,
        export_chrome_trace: bool = True,
        export_stacks: bool = False,
        tensorboard: bool = True,
        enabled: bool = True,
        rank: int = 0,
    ):
        self.output_dir = output_dir
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.repeat = repeat
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules
        self.export_chrome_trace = export_chrome_trace
        self.export_stacks = export_stacks
        self.tensorboard = tensorboard
        self.enabled = enabled
        self.rank = rank

        self._profiler: Optional[profile] = None
        self._current_step: int = 0
        self._trace_count: int = 0
        self._summary_data: List[Dict[str, Any]] = []

        if self.enabled:
            self._setup()

    def _setup(self):
        """Initialize profiler and output directory."""
        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

            # Create timestamped subdirectory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.trace_dir = os.path.join(self.output_dir, f"trace_{timestamp}")
            os.makedirs(self.trace_dir, exist_ok=True)

            # TensorBoard directory
            if self.tensorboard:
                self.tb_dir = os.path.join(self.output_dir, f"tensorboard_{timestamp}")
                os.makedirs(self.tb_dir, exist_ok=True)

    def _get_trace_handler(self):
        """Create trace handler for profiler output."""
        if not self.enabled or self.rank != 0:
            return lambda p: None

        def handler(p):
            # Export Chrome trace
            if self.export_chrome_trace:
                trace_path = os.path.join(
                    self.trace_dir, f"trace_step_{self._current_step}.json"
                )
                p.export_chrome_trace(trace_path)
                print(f"[Profiler] Chrome trace saved to {trace_path}")

            # Export stack traces
            if self.export_stacks and self.with_stack:
                stack_path = os.path.join(
                    self.trace_dir, f"stacks_step_{self._current_step}.txt"
                )
                p.export_stacks(stack_path, metric="self_cuda_time_total")
                print(f"[Profiler] Stack traces saved to {stack_path}")

            # Collect summary data
            self._collect_summary(p)
            self._trace_count += 1

        return handler

    def _get_tb_trace_handler(self):
        """Create TensorBoard trace handler."""
        if self.tensorboard and self.enabled and self.rank == 0:
            return tensorboard_trace_handler(self.tb_dir)
        return None

    def _collect_summary(self, p):
        """Collect summary statistics from profiler."""
        try:
            # Get key averages
            key_averages = p.key_averages()

            summary = {
                "step": self._current_step,
                "total_cuda_time_ms": sum(
                    e.cuda_time_total for e in key_averages if e.cuda_time_total > 0
                )
                / 1000,
                "total_cpu_time_ms": sum(
                    e.cpu_time_total for e in key_averages if e.cpu_time_total > 0
                )
                / 1000,
                "top_cuda_ops": [],
                "top_cpu_ops": [],
                "memory_stats": {},
            }

            # Top CUDA operations
            cuda_sorted = sorted(
                [e for e in key_averages if e.cuda_time_total > 0],
                key=lambda e: e.cuda_time_total,
                reverse=True,
            )[:10]

            for e in cuda_sorted:
                summary["top_cuda_ops"].append(
                    {
                        "name": e.key,
                        "cuda_time_ms": e.cuda_time_total / 1000,
                        "cuda_time_pct": (
                            e.cuda_time_total / summary["total_cuda_time_ms"] / 10
                            if summary["total_cuda_time_ms"] > 0
                            else 0
                        ),
                        "count": e.count,
                        "input_shapes": str(e.input_shapes) if e.input_shapes else "",
                    }
                )

            # Top CPU operations
            cpu_sorted = sorted(
                [e for e in key_averages if e.cpu_time_total > 0],
                key=lambda e: e.cpu_time_total,
                reverse=True,
            )[:10]

            for e in cpu_sorted:
                summary["top_cpu_ops"].append(
                    {
                        "name": e.key,
                        "cpu_time_ms": e.cpu_time_total / 1000,
                        "count": e.count,
                    }
                )

            # Memory statistics (if profiling memory)
            if self.profile_memory:
                summary["memory_stats"] = {
                    "cuda_memory_allocated_gb": (
                        torch.cuda.memory_allocated() / 1024**3
                        if torch.cuda.is_available()
                        else 0
                    ),
                    "cuda_max_memory_allocated_gb": (
                        torch.cuda.max_memory_allocated() / 1024**3
                        if torch.cuda.is_available()
                        else 0
                    ),
                    "cuda_memory_reserved_gb": (
                        torch.cuda.memory_reserved() / 1024**3
                        if torch.cuda.is_available()
                        else 0
                    ),
                }

            self._summary_data.append(summary)

        except Exception as e:
            print(f"[Profiler] Warning: Failed to collect summary: {e}")

    def start(self):
        """Start the profiler."""
        if not self.enabled:
            return

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        # Create schedule
        prof_schedule = schedule(
            wait=self.wait_steps,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=self.repeat,
        )

        # Create trace handler
        on_trace_ready = self._get_tb_trace_handler() or self._get_trace_handler()

        self._profiler = profile(
            activities=activities,
            schedule=prof_schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            with_modules=self.with_modules,
        )
        self._profiler.__enter__()
        print(
            f"[Profiler] Started profiling (wait={self.wait_steps}, "
            f"warmup={self.warmup_steps}, active={self.active_steps}, repeat={self.repeat})"
        )

    def stop(self):
        """Stop the profiler."""
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)
            self._profiler = None
            print("[Profiler] Stopped profiling")

    @contextmanager
    def step(self, step_num: int):
        """Context manager for a single training step.

        Args:
            step_num: Current step number

        Usage:
            with profiler.step(step):
                # training code
        """
        self._current_step = step_num

        if not self.enabled or self._profiler is None:
            yield
            return

        try:
            yield
        finally:
            self._profiler.step()

    def export_summary(self):
        """Export profiling summary to JSON file."""
        if not self.enabled or self.rank != 0 or not self._summary_data:
            return

        summary_path = os.path.join(self.trace_dir, "profiler_summary.json")

        # Compute aggregate statistics
        aggregate = {
            "num_profiles": len(self._summary_data),
            "avg_cuda_time_ms": (
                sum(s["total_cuda_time_ms"] for s in self._summary_data)
                / len(self._summary_data)
                if self._summary_data
                else 0
            ),
            "avg_cpu_time_ms": (
                sum(s["total_cpu_time_ms"] for s in self._summary_data)
                / len(self._summary_data)
                if self._summary_data
                else 0
            ),
            "step_details": self._summary_data,
        }

        with open(summary_path, "w") as f:
            json.dump(aggregate, f, indent=2)

        print(f"[Profiler] Summary saved to {summary_path}")
        self._print_summary(aggregate)

    def _print_summary(self, aggregate: Dict):
        """Print profiling summary to console."""
        print("\n" + "=" * 80)
        print("PROFILER SUMMARY")
        print("=" * 80)
        print(f"Number of profiled steps: {aggregate['num_profiles']}")
        print(f"Average CUDA time per step: {aggregate['avg_cuda_time_ms']:.2f} ms")
        print(f"Average CPU time per step: {aggregate['avg_cpu_time_ms']:.2f} ms")

        if self._summary_data:
            last = self._summary_data[-1]

            print("\nTop CUDA operations (last profiled step):")
            print("-" * 60)
            for op in last.get("top_cuda_ops", [])[:5]:
                print(
                    f"  {op['name'][:40]:<40} "
                    f"{op['cuda_time_ms']:>8.2f} ms ({op['count']} calls)"
                )

            if last.get("memory_stats"):
                print("\nMemory Statistics:")
                print("-" * 60)
                mem = last["memory_stats"]
                print(f"  Allocated: {mem.get('cuda_memory_allocated_gb', 0):.2f} GB")
                print(
                    f"  Max Allocated: {mem.get('cuda_max_memory_allocated_gb', 0):.2f} GB"
                )
                print(f"  Reserved: {mem.get('cuda_memory_reserved_gb', 0):.2f} GB")

        print("=" * 80 + "\n")

        # Print TensorBoard viewing instructions
        if self.tensorboard and hasattr(self, "tb_dir"):
            print(f"\nTo view detailed traces in TensorBoard, run:")
            print(f"  tensorboard --logdir={self.tb_dir}")
            print("\nIn TensorBoard, go to 'PyTorch Profiler' tab to see:")
            print("  - GPU Kernel view: SM efficiency, achieved occupancy")
            print("  - Trace view: Timeline of CPU/GPU operations")
            print("  - Memory view: Memory allocation patterns")
            print("  - Module view: Time spent in each PyTorch module\n")


def create_profiler_from_config(
    profiler_cfg: ProfilerConfig, output_dir: str, rank: int = 0
) -> TrainingProfiler:
    """Create TrainingProfiler from ProfilerConfig.

    Args:
        profiler_cfg: ProfilerConfig dataclass
        output_dir: Base output directory (profiler output will be in a subdirectory)
        rank: Distributed rank (only rank 0 writes files)

    Returns:
        TrainingProfiler instance
    """
    profiler_output = os.path.join(output_dir, "profiler")

    return TrainingProfiler(
        output_dir=profiler_output,
        wait_steps=profiler_cfg.wait_steps,
        warmup_steps=profiler_cfg.warmup_steps,
        active_steps=profiler_cfg.active_steps,
        repeat=profiler_cfg.repeat,
        record_shapes=profiler_cfg.record_shapes,
        profile_memory=profiler_cfg.profile_memory,
        with_stack=profiler_cfg.with_stack,
        with_flops=profiler_cfg.with_flops,
        with_modules=profiler_cfg.with_modules,
        export_chrome_trace=profiler_cfg.export_chrome_trace,
        export_stacks=profiler_cfg.export_stacks,
        tensorboard=profiler_cfg.tensorboard,
        enabled=profiler_cfg.enabled,
        rank=rank,
    )


# Utility function for annotating code regions
@contextmanager
def profile_region(name: str):
    """Context manager for annotating code regions in profiler traces.

    Example:
        with profile_region("data_preprocessing"):
            data = preprocess(raw_data)

        with profile_region("forward_pass"):
            output = model(data)
    """
    with record_function(name):
        yield


def get_cuda_memory_stats() -> Dict[str, float]:
    """Get current CUDA memory statistics.

    Returns:
        Dictionary with memory statistics in GB.
    """
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
    }


def print_cuda_memory_stats(prefix: str = ""):
    """Print current CUDA memory statistics."""
    stats = get_cuda_memory_stats()
    if stats:
        prefix = f"[{prefix}] " if prefix else ""
        print(
            f"{prefix}CUDA Memory - "
            f"Allocated: {stats['allocated_gb']:.2f} GB, "
            f"Max Allocated: {stats['max_allocated_gb']:.2f} GB, "
            f"Reserved: {stats['reserved_gb']:.2f} GB"
        )


def reset_peak_memory_stats():
    """Reset CUDA peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
