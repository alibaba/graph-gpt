# Graph Tokenizer Module

This module provides graph tokenization functionality for the GraphGPT project. It converts graph-structured data into token sequences suitable for transformer models.

## Architecture

The tokenizer module uses a **composition-based architecture** with clear separation of concerns:

```
BaseTokenizer (abstract)
├── GSTTokenizer          - 1D token sequences (pretrain, node/edge-level tasks)
└── StackedGSTTokenizer   - 2D stacked token sequences (graph-level tasks)

Strategies (composable components):
├── PaddingStrategy
│   ├── FlatPaddingStrategy     - For 1D sequences
│   └── StackedPaddingStrategy  - For 2D stacked sequences
├── SequencePacker              - Packs multiple short sequences
└── TaskPreparationStrategy
    ├── PretrainMLMStrategy     - MLM pre-training
    ├── PretrainCoordStrategy   - Coordinate prediction
    ├── GraphLevelStrategy      - Graph-level tasks
    ├── EdgeLevelStrategy       - Edge-level tasks
    ├── NodeLevelStrategy       - Node-level tasks
    └── NodeV2Strategy          - Token-level node classification
```

## Quick Start

### Basic Usage

```python
from src.data.tokenizer import GSTTokenizer, StackedGSTTokenizer

# For pre-training or node/edge-level tasks
tokenizer = GSTTokenizer(config)

# For graph-level tasks with stacked attributes
tokenizer = StackedGSTTokenizer(config, stack_method="short")

# Tokenize a graph
result = tokenizer(graph)
# Returns: dict with input_ids, labels, attention_mask, etc.
```

### Advanced Usage with Custom Strategies

```python
from src.data.tokenizer import BaseTokenizer
from src.data.tokenizer.strategies import (
    FlatPaddingStrategy,
    get_task_strategy,
    SequencePacker,
)

# Create custom tokenizer with specific strategies
tokenizer = BaseTokenizer(
    config,
    padding_strategy=FlatPaddingStrategy(padding_side="right"),
    task_preparer=get_task_strategy("graph")(),
    sequence_packer=SequencePacker(mpe=512, dataset=train_dataset),
)
```

## Tokenizer Classes

### GSTTokenizer

For 1D token sequences. Suitable for:
- Pre-training (MLM, CL)
- Node-level classification/regression
- Edge-level link prediction

```python
tokenizer = GSTTokenizer(
    config,
    padding_side="right",  # or "left"
    add_eos=True,
    train_cfg=training_config,
)
```

### StackedGSTTokenizer

For 2D stacked token sequences. Suitable for:
- Graph-level classification/regression
- Tasks requiring node/edge attribute stacking

```python
tokenizer = StackedGSTTokenizer(
    config,
    stack_method="short",      # or "long"
    rotation="anchor_rotate",  # or "trans_rotate"
    padding_side="right",
    add_eos=True,
)
```

## Strategy Classes

### Padding Strategies

```python
from src.data.tokenizer.strategies import FlatPaddingStrategy, StackedPaddingStrategy

# For 1D sequences
padding = FlatPaddingStrategy(
    pad_token_id=0,
    label_pad_token_id=-100,
    padding_side="right",
)

# For 2D stacked sequences
padding = StackedPaddingStrategy(
    pad_token_id=0,
    label_pad_token_id=-100,
    padding_side="right",
)

# Pad a batch
padded = padding.pad_batch(features, max_length=128)
```

### Sequence Packer

Packs multiple short sequences into one long sequence for efficient training:

```python
from src.data.tokenizer.strategies import SequencePacker

packer = SequencePacker(
    mpe=512,                    # Max position embeddings
    dataset=train_dataset,
    sampler=None,               # Optional sampler
    random_ratio=1.0,           # Ratio of random vs sequential sampling
    eos_token="<eos>",
    label_pad_token="<label_pad>",
)

# Use with tokenizer
tokenizer.sequence_packer = packer
```

### Task Preparation Strategies

```python
from src.data.tokenizer.strategies import get_task_strategy

# Available task types:
# - "pretrain", "pretrain-mlm", "pretrain-cl"
# - "pretrain-coord", "pretrain-smtp"
# - "graph" (graph-level)
# - "edge" (edge-level)
# - "node" (node-level)
# - "nodev2" (token-level node)

strategy = get_task_strategy("graph")
prepared_inputs = strategy.prepare(in_dict, token_res, graph, tokenizer)
```

## Configuration

Example configuration structure:

```python
config = {
    "name_or_path": "path/to/vocab",
    "vocab_file": "vocab",
    "task_type": "graph",  # or "pretrain-mlm", "node", "edge", etc.
    "structure": {
        "node": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "node_scope": 100000,
        },
        "edge": {
            "bi_token": "<edge>",
            "jump_token": "<jump>",
            "remove_edge_type_token": True,
        },
        "graph": {
            "summary_token": "<gsum>",
        },
        "common": {
            "mask_token": "<mask>",
            "icl_token": "<icl>",
            "sep_token": "<sep>",
        },
    },
    "semantics": {
        "attr_assignment": "random",
        "attr_shuffle": False,
        "node": {"dim": 9},
        "edge": {"dim": 3},
    },
}
```

## Output Format

The tokenizer returns a dictionary with:

```python
{
    "input_ids": [...],        # Token IDs
    "labels": [...],           # Label token IDs
    "attention_mask": [...],   # 1 for real tokens, 0 for padding
    "position_ids": [...],     # (optional) Position IDs
    "split_lens": [...],       # (optional) Lengths for packed sequences
    "attn_modes": [...],       # (optional) Attention modes
    # Task-specific fields:
    "graph_labels": [...],     # For graph-level tasks
    "node_labels": [...],      # For node-level tasks
    "edge_labels": [...],      # For edge-level tasks
}
```

## Backward Compatibility

All existing imports continue to work:

```python
# Old imports still work
from src.data.tokenizer import GSTTokenizer, StackedGSTTokenizer
from src.data.tokenizer import get_input_dict_from_seq_tokens_id
from src.data.tokenizer import DICT_pos_func
```

## Testing

Run tests:

```bash
# Syntax and structure tests
pytest tests/test_refactoring_syntax.py -v

# Full smoke tests (requires dependencies)
pytest tests/test_tokenizer_smoke.py -v
```

## Migration Guide

### From Old Monolithic Tokenizer

**Before:**
```python
from src.data.tokenizer import GSTTokenizer

tokenizer = GSTTokenizer(config)
tokenizer.mpe = 512  # For packing
tokenizer.dataset = train_dataset
```

**After:**
```python
from src.data.tokenizer import GSTTokenizer
from src.data.tokenizer.strategies import SequencePacker

tokenizer = GSTTokenizer(config)
tokenizer.setup_sequence_packing(mpe=512, dataset=train_dataset)
```

### Using Custom Strategies

```python
from src.data.tokenizer import BaseTokenizer
from src.data.tokenizer.strategies import (
    FlatPaddingStrategy,
    get_task_strategy,
)

# Create with custom strategies
tokenizer = BaseTokenizer(
    config,
    padding_strategy=FlatPaddingStrategy(padding_side="left"),
    task_preparer=get_task_strategy("node")(),
)
```

## File Structure

```
src/data/tokenizer/
├── __init__.py              # Public exports
├── base.py                  # BaseTokenizer (ABC)
├── core.py                  # GSTTokenizer, StackedGSTTokenizer
├── types.py                 # TokenizationOutput dataclass
├── vocab_builder.py         # Vocabulary building utilities
├── padding.py               # Padding utilities
├── stacking.py              # Attribute stacking utilities
├── graph_encoding.py        # Graph encoding utilities
├── masking.py               # Masking utilities
└── strategies/              # Strategy pattern implementations
    ├── __init__.py
    ├── padding.py
    ├── packing.py
    └── task_prep/
        ├── __init__.py
        ├── base.py
        ├── pretrain.py
        └── supervised.py
```

## License

Same as the main GraphGPT project.
