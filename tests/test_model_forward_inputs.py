#!/usr/bin/env python3
"""
Test script to print model forward inputs under different training configurations.

This script inspects what data is passed to the model's forward() method during:
1. Pre-training mode
2. Fine-tuning mode
3. Different configurations (DeepSpeed vs DDP, token packing, etc.)

Usage:
    # Test pre-training configuration
    python tests/test_model_forward_inputs.py --mode pretrain --config configs/config.yaml

    # Test fine-tuning configuration
    python tests/test_model_forward_inputs.py --mode finetune --config configs/config.yaml

    # Test with specific dataset
    python tests/test_model_forward_inputs.py --mode pretrain --dataset ogbg_molpcba
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf
from hydra import initialize, compose

from src.conf import Config
from src.training import TrainingPipeline
from src.training.pretrain_mode import PretrainMode
from src.training.finetune_mode import FinetuneMode
from src.data import collator


class ForwardInputInspector:
    """Hook-based inspector to capture model forward() inputs."""

    def __init__(self):
        self.captured_inputs = []
        self.original_forward = None
        self.model = None

    def attach(self, model):
        """Attach hook to model's forward method."""
        self.model = model
        self.original_forward = model.forward

        def wrapped_forward(*args, **kwargs):
            # Store a copy of inputs
            captured = {
                "args": args,
                "kwargs": kwargs.copy(),
            }
            self.captured_inputs.append(captured)

            # Print the inputs
            self._print_inputs(captured)

            # Call original forward
            return self.original_forward(*args, **kwargs)

        model.forward = wrapped_forward
        return self

    def detach(self):
        """Remove hook from model."""
        if self.original_forward:
            self.model.forward = self.original_forward

    def _print_inputs(self, captured):
        """Print captured forward inputs in a readable format."""
        print("\n" + "=" * 80)
        print("MODEL FORWARD INPUTS")
        print("=" * 80)

        # Print positional args
        if captured["args"]:
            print(f"\nPositional arguments ({len(captured['args'])}):")
            for i, arg in enumerate(captured["args"]):
                if isinstance(arg, torch.Tensor):
                    print(
                        f"  [{i}] Tensor: shape={arg.shape}, dtype={arg.dtype}, device={arg.device}"
                    )
                    print(
                        f"      min={arg.min().item():.4f}, max={arg.max().item():.4f}"
                    )
                    if arg.numel() < 20:
                        print(f"      values={arg.flatten().tolist()}")
                else:
                    print(f"  [{i}] {type(arg).__name__}: {arg}")

        # Print keyword args
        if captured["kwargs"]:
            print(f"\nKeyword arguments ({len(captured['kwargs'])}):")
            for key, value in captured["kwargs"].items():
                if isinstance(value, torch.Tensor):
                    print(
                        f"  {key}: shape={value.shape}, dtype={value.dtype}, device=value.device"
                    )
                    if value.numel() > 0:
                        print(
                            f"        min={value.min().item():.4f}, max={value.max().item():.4f}"
                        )
                        if value.numel() < 20:
                            print(f"        values={value.flatten().tolist()}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {type(value).__name__} with {len(value)} items")
                    if len(value) > 0 and len(value) < 5:
                        for i, item in enumerate(value):
                            if isinstance(item, torch.Tensor):
                                print(f"    [{i}] Tensor: shape={item.shape}")
                            else:
                                print(f"    [{i}] {type(item).__name__}: {item}")
                else:
                    print(f"  {key}: {type(value).__name__} = {value}")

        print("=" * 80 + "\n")


def setup_config(mode: str, config_overrides: list = None):
    """Setup configuration for testing."""
    print(f"\n{'='*80}")
    print(f"SETTING UP {mode.upper()} CONFIGURATION")
    print(f"{'='*80}")

    if config_overrides is None:
        config_overrides = []

    # Load base config
    try:
        cfg = compose(
            config_name="config",
            overrides=config_overrides,
        )
        cfg = OmegaConf.to_object(cfg)
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Creating minimal test config...")
        cfg = Config()

    # Apply mode-specific settings
    if mode == "pretrain":
        cfg.training.task_type = "pretrain"
        cfg.training.pretrain_mode = True
        cfg.training.batch_size = 2  # Small batch for testing
        cfg.training.num_workers = 0  # Disable multiprocessing for debugging
        cfg.training.num_workers_eval = 0
        cfg.tokenization.data.dataset = "ogbg_molpcba"  # Use small test dataset
        cfg.training.output_dir = "./tmp_test_pretrain"
        cfg.training.pack_tokens = 0  # Disable token packing for clarity
        cfg.model.max_position_embeddings = 512

    elif mode == "finetune":
        cfg.training.task_type = "finetune"
        cfg.training.pretrain_mode = False
        cfg.training.batch_size = 2
        cfg.training.num_workers = 0
        cfg.training.num_workers_eval = 0
        cfg.tokenization.data.dataset = "ogbg_molpcba"
        cfg.training.output_dir = "./tmp_test_finetune"
        cfg.training.max_length = 512
        cfg.model.ft_head.num_labels = 1

    # Disable DeepSpeed for simpler testing
    cfg.training.use_deepspeed = False
    cfg.training.deepspeed_conf_file = ""

    print(f"Configuration loaded successfully")
    print(f"Mode: {mode}")
    print(f"Dataset: {cfg.tokenization.data.dataset}")
    print(f"Batch size: {cfg.training.batch_size}")

    return cfg


def test_pretrain_forward():
    """Test pre-training forward inputs."""
    print("\n" + "=" * 80)
    print("TEST PRE-TRAINING FORWARD INPUTS")
    print("=" * 80)

    cfg = setup_config(
        "pretrain",
        [
            "training.task_type=pretrain",
            "training.batch_size=2",
            "training.num_workers=0",
            "tokenization.data.dataset=ogbg_molpcba",
            "training.output_dir=./tmp_test_pretrain",
            "training.pack_tokens=0",
            "model.max_position_embeddings=512",
        ],
    )

    # Create pipeline
    pipeline = TrainingPipeline(cfg, PretrainMode())

    # Attach inspector before running
    inspector = ForwardInputInspector()

    # Monkey-patch the model creation to attach inspector
    original_create_model = pipeline._create_model

    def patched_create_model():
        original_create_model()
        inspector.attach(pipeline.model)

    pipeline._create_model = patched_create_model

    # Run setup phases
    try:
        pipeline._extract_config()
        pipeline.mode.update_config(pipeline)
        pipeline._create_ema_config()
        pipeline._setup_deepspeed_flag()
        pipeline._setup_distributed()
        pipeline._init_data_configs()
        pipeline.mode.prepare_data(pipeline)
        pipeline._create_model()
        pipeline.mode.setup_optimizer(pipeline)
        pipeline.mode.setup_training(pipeline)

        # Get one batch from train loader
        print("\nFetching one batch from training loader...")
        for i, data in enumerate(pipeline.mode.train_loader):
            if i >= 1:
                break
            print(f"\nBatch {i} structure:")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {type(value).__name__} with {len(value)} items")
                else:
                    print(f"  {key}: {type(value).__name__} = {value}")

            # Call training step (which calls forward)
            print("\nCalling training step (forward pass)...")
            from src.utils import training_utils

            training_utils.batch_training(
                data,
                pipeline.model,
                pipeline.train_cfg,
                pipeline.train_stats,
                pipeline.opt_stats,
            )
            break

    except Exception as e:
        print(f"\nError during pre-training test: {e}")
        import traceback

        traceback.print_exc()
    finally:
        inspector.detach()


def test_finetune_forward():
    """Test fine-tuning forward inputs."""
    print("\n" + "=" * 80)
    print("TEST FINE-TUNING FORWARD INPUTS")
    print("=" * 80)

    cfg = setup_config(
        "finetune",
        [
            "training.task_type=finetune",
            "training.batch_size=2",
            "training.num_workers=0",
            "tokenization.data.dataset=ogbg_molpcba",
            "training.output_dir=./tmp_test_finetune",
            "training.max_length=512",
            "model.ft_head.num_labels=1",
        ],
    )

    # Create pipeline
    pipeline = TrainingPipeline(cfg, FinetuneMode())

    # Attach inspector
    inspector = ForwardInputInspector()

    # Monkey-patch model creation
    original_create_model = pipeline._create_model

    def patched_create_model():
        original_create_model()
        inspector.attach(pipeline.model)

    pipeline._create_model = patched_create_model

    # Run setup phases
    try:
        pipeline._extract_config()
        pipeline.mode.update_config(pipeline)
        pipeline._create_ema_config()
        pipeline._setup_deepspeed_flag()
        pipeline._setup_distributed()
        pipeline._init_data_configs()
        pipeline.mode.prepare_data(pipeline)
        pipeline._create_model()
        pipeline.mode.setup_optimizer(pipeline)
        pipeline.mode.setup_training(pipeline)

        # Get one batch from train loader
        print("\nFetching one batch from training loader...")
        for i, data in enumerate(pipeline.mode.train_loader_for_eval):
            if i >= 1:
                break
            print(f"\nBatch {i} structure:")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {type(value).__name__} with {len(value)} items")
                else:
                    print(f"  {key}: {type(value).__name__} = {value}")

            # Call training step
            print("\nCalling training step (forward pass)...")
            from src.utils import training_utils

            training_utils.ft_batch_training(
                data,
                pipeline.model,
                pipeline.model_cfg.ft_head,
                pipeline.train_cfg,
                pipeline.train_stats,
                pipeline.opt_stats,
            )
            break

    except Exception as e:
        print(f"\nError during fine-tuning test: {e}")
        import traceback

        traceback.print_exc()
    finally:
        inspector.detach()


def test_collator_output():
    """Test collator output structure directly."""
    print("\n" + "=" * 80)
    print("TEST COLLATOR OUTPUT STRUCTURE")
    print("=" * 80)

    from src.data import tokenizer, vocab_builder, read_dataset
    from src.utils import conf_utils, inspect_tokenization_results

    # Setup minimal config
    cfg = setup_config("pretrain")

    # Build tokenizer
    tokenizer_config = conf_utils.convert_to_legacy_tokenization_config(cfg)
    dataset, raw_dataset = read_dataset(
        name=cfg.tokenization.data.dataset,
        data_cfg=cfg.tokenization.data,
        train_cfg=cfg.training,
    )

    # Build vocab
    vocab_builder.build_vocab(raw_dataset, tokenizer_config, 0)

    # Init tokenizer
    tokenizer_cls = getattr(tokenizer, tokenizer_config["tokenizer_class"])
    gtokenizer = tokenizer_cls(
        tokenizer_config,
        add_eos=cfg.tokenization.add_eos,
        stack_method=cfg.model.graph_input.stack_method,
        train_cfg=cfg.training,
    )

    # Create collator
    collator_fn = collator.DataCollatorForGST(
        tokenizer=gtokenizer,
        max_length=cfg.model.max_position_embeddings,
        pad_to_multiple_of=cfg.training.pad_to_multiple_of,
        return_tensors="pt",
        is_training=True,
    )

    # Get sample graphs
    sample_graphs = []
    for i in range(2):
        if hasattr(dataset, "__getitem__"):
            graph = dataset[i]
        else:
            graph = next(iter(dataset))
        sample_graphs.append(graph)

    # Collate
    print("\nCollating sample graphs...")
    batch = collator_fn(sample_graphs)

    print("\nCollated batch structure:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            print(f"      min={value.min().item():.4f}, max={value.max().item():.4f}")
        elif isinstance(value, (list, tuple)):
            print(f"  {key}: {type(value).__name__} with {len(value)} items")
            if len(value) > 0:
                print(f"      first item type: {type(value[0]).__name__}")
        else:
            print(f"  {key}: {type(value).__name__}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "finetune", "collator", "all"],
        default="all",
        help="Which mode to test",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file (optional)"
    )
    parser.add_argument(
        "--dataset", type=str, default="ogbg_molpcba", help="Dataset to use for testing"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("MODEL FORWARD INPUT INSPECTION TEST")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print("=" * 80)

    if args.mode in ["pretrain", "all"]:
        test_pretrain_forward()

    if args.mode in ["finetune", "all"]:
        test_finetune_forward()

    if args.mode in ["collator", "all"]:
        test_collator_output()

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
