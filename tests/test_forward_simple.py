#!/usr/bin/env python3
"""
Simple test script to print model forward inputs.

This script shows what data is passed to model.forward() during training.

Usage:
    python tests/test_forward_simple.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from src.conf import Config
from src.data import tokenizer, vocab_builder, read_dataset, collator
from src.utils import conf_utils
from src.models import GraphGPTPretrainBase, GraphGPTTaskModel, convert_to_legacy_config
from src.utils import modules_utils


def print_tensor_info(name, tensor):
    """Print detailed info about a tensor."""
    if not isinstance(tensor, torch.Tensor):
        print(f"{name}: {type(tensor).__name__} = {tensor}")
        return

    info = f"{name}: shape={list(tensor.shape)}, dtype={tensor.dtype}"
    if tensor.numel() > 0:
        info += f", min={tensor.min().item():.4f}, max={tensor.max().item():.4f}"
        if tensor.numel() <= 10:
            info += f", values={tensor.flatten().tolist()}"
    print(info)


def test_pretrain_forward():
    """Test pre-training forward pass with dummy data."""
    print("\n" + "=" * 80)
    print("PRE-TRAINING FORWARD PASS TEST")
    print("=" * 80)

    # Create minimal config using OmegaConf to properly handle nested structures
    from omegaconf import OmegaConf
    from src.conf.base_configs import Config as BaseConfig

    # Start with empty config and override
    cfg_dict = {
        "tokenization": {
            "attr_world_identifier": "@",
            "vocab_file": "dummy_vocab.txt",
            "label_tokens_to_pad": [],
            "semantics": {
                "node": {
                    "discrete": None,
                    "dim": 0,
                    "continuous": None,
                    "ignored_val": None,
                    "embed": None,
                    "embed_dim": 0,
                },
                "edge": {
                    "discrete": None,
                    "dim": 0,
                    "continuous": None,
                    "ignored_val": None,
                    "embed": None,
                    "embed_dim": 0,
                },
                "graph": {
                    "discrete": None,
                    "dim": 0,
                    "continuous": None,
                    "ignored_val": None,
                    "embed": None,
                    "embed_dim": 0,
                },
                "common": {"reserved_token": ["<pad>", "<mask>"], "numbers": []},
                "instructions": {"enable": False, "name": "none", "func": []},
            },
            "structure": {
                "nx": {"enable": True, "func": []},
                "node": {
                    "bos_token": "<bos>",
                    "eos_token": "<eos>",
                    "new_node_token": "<new_node>",
                    "node_scope": 1,
                    "cyclic": 0,
                },
                "edge": {
                    "remove_edge_type_token": False,
                    "in_token": "<in>",
                    "out_token": "<out>",
                    "bi_token": "<bi>",
                    "jump_token": "<jump>",
                },
                "graph": {"summary_token": "<summary>"},
                "common": {
                    "mask_token": "<mask>",
                    "icl_token": "<icl>",
                    "sep_token": "<sep>",
                    "reserved_token": ["<pad>", "<mask>"],
                },
            },
        },
        "training": {
            "task_type": "pretrain",
            "batch_size": 2,
            "pretrain_mlm": {
                "name": "mlm",
                "params": {
                    "fixed_ratio": 0.15,
                    "power": 1,
                    "mtp": [3],
                    "umr_clip": [0.0, 1.0],
                },
            },
        },
        "model": {
            "max_position_embeddings": 128,
            "graph_input": {"stack_method": "sum", "stacked_feat": 1},
            "embed_dim": 0,
            "vocab_size": 512,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "rms_norm_eps": 1e-5,
            "initializer_range": 0.02,
        },
    }

    # Convert to OmegaConf and then to dataclass
    cfg_omega = OmegaConf.create(cfg_dict)
    cfg = OmegaConf.merge(OmegaConf.structured(BaseConfig), cfg_omega)
    cfg = OmegaConf.to_object(cfg)

    # Create model
    print("\nCreating model...")
    model_cfg = convert_to_legacy_config(cfg.model)
    model = GraphGPTPretrainBase(model_cfg)
    model.train()

    # Create dummy input (simulating batched tokenized graphs)
    batch_size = 2
    seq_len = 64
    stacked_feat = cfg.model.graph_input.stacked_feat

    print(f"\nCreating dummy batch:")
    print(f"  batch_size={batch_size}, seq_len={seq_len}, stacked_feat={stacked_feat}")

    # Simulate tokenized input: [batch, seq, stacked_feat]
    input_ids = torch.randint(
        0, cfg.model.vocab_size, (batch_size, seq_len, stacked_feat)
    )
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, cfg.model.vocab_size, (batch_size, seq_len, 1))

    # Some labels set to -100 (ignored in loss)
    labels[:, ::2, :] = -100  # Mask every other position

    print("\nInput tensors:")
    print_tensor_info("input_ids", input_ids)
    print_tensor_info("attention_mask", attention_mask)
    print_tensor_info("labels", labels)

    # Forward pass
    print("\nCalling model.forward()...")
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    print("\nOutput:")
    if hasattr(output, "head1_loss"):
        print(f"  head1_loss: {output.head1_loss}")
    if hasattr(output, "logits"):
        print_tensor_info("  logits", output.logits)

    print("\n" + "=" * 80)


def test_finetune_forward():
    """Test fine-tuning forward pass with dummy data."""
    print("\n" + "=" * 80)
    print("FINE-TUNING FORWARD PASS TEST")
    print("=" * 80)

    # Create minimal config
    cfg = Config()
    cfg.training.task_type = "finetune"
    cfg.training.batch_size = 2
    cfg.model.max_position_embeddings = 128
    cfg.model.graph_input.stack_method = "sum"
    cfg.model.graph_input.stacked_feat = 1
    cfg.tokenization.add_eos = True
    cfg.model.embed_dim = 0
    cfg.model.vocab_size = 512
    cfg.model.hidden_size = 64
    cfg.model.num_hidden_layers = 2
    cfg.model.num_attention_heads = 4
    cfg.model.intermediate_size = 128
    cfg.model.rms_norm_eps = 1e-5
    cfg.model.initializer_range = 0.02
    cfg.model.ft_head.num_labels = 1
    cfg.model.ft_head.loss_type = "regression"

    # Create model
    print("\nCreating model...")
    model_cfg = convert_to_legacy_config(cfg.model)
    model = GraphGPTTaskModel(model_cfg)
    model.train()

    # Create dummy input
    batch_size = 2
    seq_len = 64
    stacked_feat = cfg.model.graph_input.stacked_feat

    print(f"\nCreating dummy batch:")
    print(f"  batch_size={batch_size}, seq_len={seq_len}, stacked_feat={stacked_feat}")

    input_ids = torch.randint(
        0, cfg.model.vocab_size, (batch_size, seq_len, stacked_feat)
    )
    attention_mask = torch.ones(batch_size, seq_len)
    task_labels = torch.randn(batch_size, 1)  # Regression target

    print("\nInput tensors:")
    print_tensor_info("input_ids", input_ids)
    print_tensor_info("attention_mask", attention_mask)
    print_tensor_info("task_labels", task_labels)

    # Forward pass
    print("\nCalling model.forward()...")
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_labels=task_labels,
        )

    print("\nOutput:")
    if hasattr(output, "task_loss"):
        print(f"  task_loss: {output.task_loss}")
    if hasattr(output, "task_logits"):
        print_tensor_info("  task_logits", output.task_logits)

    print("\n" + "=" * 80)


def test_collator_batch_structure():
    """Test how collator creates batches from graph samples."""
    print("\n" + "=" * 80)
    print("COLLATOR BATCH STRUCTURE TEST")
    print("=" * 80)

    # Try to load a real dataset if available
    try:
        from src.conf import base_configs

        cfg = Config()
        cfg.tokenization.data.dataset = "ogbg_molpcba"
        cfg.tokenization.data.return_valid_test = False
        cfg.training.batch_size = 2
        cfg.training.valid_percent = 0

        print(f"\nLoading dataset: {cfg.tokenization.data.dataset}")
        dataset, raw_dataset = read_dataset(
            name=cfg.tokenization.data.dataset,
            data_cfg=cfg.tokenization.data,
            train_cfg=cfg.training,
        )

        # Build vocab
        tokenizer_config = conf_utils.convert_to_legacy_tokenization_config(cfg)
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

        print(f"\nCollating {len(sample_graphs)} graphs...")
        batch = collator_fn(sample_graphs)

        print("\nBatch structure after collation:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={list(value.shape)}, dtype={value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"  {key}: {type(value).__name__}")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\nError loading real dataset: {e}")
        print("Skipping real data test.")
        print("\n" + "=" * 80)


def main():
    print("\n" + "=" * 80)
    print("MODEL FORWARD INPUT INSPECTION")
    print("=" * 80)

    # Test 1: Pre-training with dummy data
    test_pretrain_forward()

    # Test 2: Fine-tuning with dummy data
    test_finetune_forward()

    # Test 3: Real collator output (if dataset available)
    test_collator_batch_structure()

    print("\nAll tests completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
