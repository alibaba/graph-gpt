#!/usr/bin/env python3
"""
Minimal test showing model forward input structure.

This script prints what tensors are passed to model.forward() without requiring
full environment setup.

Usage:
    python tests/test_forward_minimal.py
"""


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def describe_tensor(name, shape, dtype="float32", purpose="", example_values=None):
    """Print description of a tensor."""
    print(f"{name}:")
    print(f"  Shape: {shape}")
    print(f"  Dtype: {dtype}")
    if purpose:
        print(f"  Purpose: {purpose}")
    if example_values:
        print(f"  Example values: {example_values}")
    print()


print_section("MODEL FORWARD INPUTS - TRAINING CONFIGURATIONS")

# ============================================================================
# 1. PRE-TRAINING MODE
# ============================================================================
print_section("1. PRE-TRAINING MODE (GraphGPTPretrainBase)")

print("Configuration:")
print("  - Task: Next token prediction / Masked token prediction")
print("  - Model: GraphGPTPretrainBase")
print("  - Batch size: Typically 32-512")
print("  - Sequence length: Up to max_position_embeddings (e.g., 1024)")
print()

print("Forward Method Signature:")
print("""
def forward(
    self,
    input_ids: torch.LongTensor = None,           # REQUIRED
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    inputs_raw_embeds: Optional[torch.FloatTensor] = None,  # For node/edge embeddings
    labels: Optional[torch.LongTensor] = None,    # For MLM/SMTP objective
    label_mask: Optional[torch.Tensor] = None,
    sample_wgt: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[bool] = None,
    split_lens=None,        # For flex attention (graph boundaries)
    attn_modes=None,        # For flex attention (attention type per graph)
) -> Union[Tuple, DoubleHeadsModelOutput]
""")

print("Typical Input Tensors (Training Mode):")
print()

describe_tensor(
    "input_ids",
    shape="[batch_size, seq_len, stacked_feat]",
    dtype="int64",
    purpose="Token IDs for each position. stacked_feat=1 typically.",
    example_values="[[[123], [456], [789]], ...]"
)

describe_tensor(
    "attention_mask",
    shape="[batch_size, seq_len]",
    dtype="float32",
    purpose="1 for real tokens, 0 for padding. Used in attention calculation.",
    example_values="[1, 1, 1, 0, 0] (first 3 real, last 2 padded)"
)

describe_tensor(
    "labels",
    shape="[batch_size, seq_len, 1] or [batch_size, seq_len, next_n_token]",
    dtype="int64",
    purpose="Target tokens for prediction. -100 means ignore in loss.",
    example_values="[[[-100], [456], [-100], [789]], ...] (alternating mask)"
)

describe_tensor(
    "inputs_raw_embeds",
    shape="[batch_size, seq_len, embed_dim]",
    dtype="float32",
    purpose="Raw node/edge embeddings (if enabled). Added to token embeddings.",
    example_values="Continuous vectors from graph features"
)

describe_tensor(
    "position_ids",
    shape="[batch_size, seq_len]",
    dtype="int64",
    purpose="Position indices [0, 1, 2, ..., seq_len-1]. Optional if sequential.",
    example_values="[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]"
)

print("Special Inputs for Advanced Features:")
print()

describe_tensor(
    "split_lens",
    shape="List[int] (not a tensor)",
    purpose="Length of each graph in batch (for variable-length packing).",
    example_values="[64, 128, 96, 32] (4 graphs with different lengths)"
)

describe_tensor(
    "attn_modes",
    shape="List[str] (not a tensor)",
    purpose="Attention mode per graph: 'flash', 'memory_efficient', etc.",
    example_values="['flash', 'flash', 'sdpa', 'flex']"
)

print("Output Structure:")
print("""
DoubleHeadsModelOutput(
    head1_loss: Tensor      # Main pre-training loss (generative/discriminative)
    head2_loss: Tensor      # Auxiliary loss (if enabled, e.g., SMTP)
    logits: Tensor          # Token prediction logits [batch, seq, vocab_size]
    hidden_states: Tuple    # Hidden states from all layers (if requested)
)
""")

# ============================================================================
# 2. FINE-TUNING MODE
# ============================================================================
print_section("2. FINE-TUNING MODE (GraphGPTTaskModel)")

print("Configuration:")
print("  - Task: Graph classification/regression")
print("  - Model: GraphGPTTaskModel or GraphGPTDenoisingRegressionDoubleHeadsModel")
print("  - Batch size: Typically 16-128")
print("  - Sequence length: Variable, up to max_length")
print()

print("Forward Method Signature (GraphGPTTaskModel):")
print("""
def forward(
    self,
    input_ids: torch.LongTensor = None,           # REQUIRED
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    inputs_raw_embeds: Optional[torch.FloatTensor] = None,
    task_labels: Optional[torch.LongTensor] = None,  # Classification targets
    cls_idx: Optional[torch.LongTensor] = None,      # [batch] pooling index
    sample_wgt: Optional[torch.FloatTensor] = None,  # Sample weights
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    split_lens=None,
    attn_modes=None,
) -> Union[Tuple, DoubleHeadsModelOutput]
""")

print("Typical Input Tensors (Training Mode):")
print()

describe_tensor(
    "input_ids",
    shape="[batch_size, seq_len, stacked_feat]",
    dtype="int64",
    purpose="Same as pre-training. Token IDs from serialized graphs.",
)

describe_tensor(
    "attention_mask",
    shape="[batch_size, seq_len]",
    dtype="float32",
    purpose="Same as pre-training. Masks padding positions.",
)

describe_tensor(
    "task_labels",
    shape="[batch_size, num_labels] or [batch_size, 1]",
    dtype="float32 (regression) or int64 (classification)",
    purpose="Ground truth labels for downstream task.",
    example_values="Regression: [[0.523], [-0.102], [1.456]]\nClassification: [[0], [1], [0]]"
)

describe_tensor(
    "cls_idx",
    shape="[batch_size]",
    dtype="int64",
    purpose="Index of [CLS] token or pooling position. Used for readout.",
    example_values="[63, 127, 95, 31] (last real token position per graph)"
)

describe_tensor(
    "sample_wgt",
    shape="[batch_size]",
    dtype="float32",
    purpose="Weight for each sample in loss. Handles class imbalance.",
    example_values="[1.0, 2.5, 0.8, 1.2] (higher weight for rare classes)"
)

print("For Denoising Models (GraphGPTDenoisingRegressionDoubleHeadsModel):")
print()

describe_tensor(
    "pretrain_labels",
    shape="[batch_size, seq_len, 1]",
    dtype="float32",
    purpose="Auxiliary pre-training target (denoising autoencoder).",
    example_values="Original node positions before noise added"
)

describe_tensor(
    "delta_pos",
    shape="[batch_size, seq_len, 3]",
    dtype="float32",
    purpose="Noise added to positions for denoising task.",
    example_values="Small perturbations [[0.01, -0.02, 0.03], ...]"
)

print("Output Structure:")
print("""
DoubleHeadsModelOutput(
    pretrain_loss: Tensor   # Auxiliary pre-training loss (if enabled)
    task_loss: Tensor       # Main fine-tuning loss
    pretrain_logits: Tensor # Auxiliary predictions (if enabled)
    task_logits: Tensor     # Task predictions [batch, num_labels]
    hidden_states: Tuple    # Hidden states (if requested)
)
""")

# ============================================================================
# 3. DATA FLOW: FROM GRAPHS TO MODEL INPUTS
# ============================================================================
print_section("3. DATA FLOW: HOW GRAPHS BECOME TENSORS")

print("Step 1: Raw Graph Data")
print("""
PyG Data object:
  - x: Node features [num_nodes, num_node_features]
  - edge_index: Connectivity [2, num_edges]
  - edge_attr: Edge features [num_edges, num_edge_features]
  - y: Graph label (for fine-tuning)
""")

print("Step 2: Tokenization (Eulerian Path)")
print("""
Graph -> Eulerian path traversal -> Sequence of node/edge tokens
Each token contains:
  - Structural info: node_id, edge_type, position
  - Semantic info: atom type, bond type, features
  
Example sequence:
  [NODE_0, EDGE_0_1, NODE_1, EDGE_1_2, NODE_2, ...]
""")

print("Step 3: Attribute Stacking")
print("""
If stacked_feat > 1, consecutive tokens are stacked:
  
  Original:  [(node0), (edge0), (node1), (edge1), ...]
  Stacked:   [[node0, edge0], [node1, edge1], ...]
  
Methods: 'sum', 'cat', 'interleave'
""")

print("Step 4: Collation (Batching)")
print("""
DataCollatorForGST:
  1. Tokenizes each graph in batch
  2. Pads sequences to same length
  3. Creates attention masks
  4. Prepares labels
  5. Adds special tokens ([BOS], [EOS], [PAD])
  
Result: Batch dict with keys:
  {
    'input_ids': Tensor[batch, seq, feat],
    'attention_mask': Tensor[batch, seq],
    'labels': Tensor[batch, seq, 1],
    'idx': List[int]  # Original graph indices
  }
""")

print("Step 5: Model Forward Pass")
print("""
Batch -> Model.forward() -> Loss

The collated batch is unpacked and passed to model:
  model(**batch)  # ** unpacks dict to keyword args
  
Inside forward():
  1. Convert input_ids to embeddings
  2. Add raw embeddings (if provided)
  3. Apply positional encoding
  4. Process through transformer layers
  5. Calculate logits and loss
""")

# ============================================================================
# 4. CONFIGURATION IMPACT ON INPUTS
# ============================================================================
print_section("4. HOW CONFIG AFFECTS FORWARD INPUTS")

print("Token Packing (training.pack_tokens > 0):")
print("""
Without packing:
  input_ids: [batch_size, seq_len, feat]
  Each sequence is one graph
  
With packing (pack_tokens=0.5):
  input_ids: [1, packed_seq_len, feat]
  Multiple graphs packed into one long sequence
  split_lens indicates boundaries: [64, 128, 96] means 3 graphs
  
Benefit: Better GPU utilization, less padding
""")

print("Flex Attention:")
print("""
Enables dynamic attention masks for variable-length graphs.

Inputs needed:
  - split_lens: Graph lengths in batch
  - attn_modes: Attention type per graph
  
Effect: Instead of global causal mask, creates block-diagonal mask
where each block is one graph's attention.
""")

print("Node/Edge Embeddings (model.embed_dim > 0):")
print("""
Adds inputs_raw_embeds to forward pass:
  
  inputs_raw_embeds: [batch, seq, embed_dim]
  
These are projected and added to token embeddings:
  inputs_embeds = token_embed(input_ids) + embed_proj(inputs_raw_embeds)
  
Use case: Continuous node features (atom coordinates, molecular properties)
""")

print("DeepSpeed vs DDP:")
print("""
DeepSpeed:
  - May use gradient accumulation
  - FP16/BF16 mixed precision
  - Distributed across GPUs
  
DDP:
  - Standard PyTorch distributed
  - Usually FP32 or manual AMP
  
Forward inputs are the same, but internal dtypes may differ.
""")

# ============================================================================
# 5. EXAMPLE BATCH INSPECTION CODE
# ============================================================================
print_section("5. HOW TO INSPECT FORWARD INPUTS IN YOUR CODE")

print("Method 1: Print in Training Loop")
print("""
# In your training script (e.g., training_utils.py)
def batch_training(data, model, train_cfg, train_stats, opt_stats):
    # Print batch structure
    print("\\n=== BATCH INSPECTION ===")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}, "
                  f"min={value.min():.4f}, max={value.max():.4f}")
    print("========================\\n")
    
    # Then proceed with forward pass
    output = model(**data)
    ...
""")

print("Method 2: Hook on Forward Method")
print("""
class ForwardInspector:
    def __init__(self):
        self.captured = []
    
    def hook(self, module, args, kwargs):
        self.captured.append({
            'args': args,
            'kwargs': kwargs
        })
        # Print info
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")

# Usage:
inspector = ForwardInspector()
model.register_forward_hook(inspector.hook)

# Run training...
# Inspector will capture all forward calls
""")

print("Method 3: Modify Collator")
print("""
# In src/data/collator.py, add debug prints:
def __call__(self, graphs, return_tensors=None):
    features = [...]  # Tokenize graphs
    
    batch = self.tokenizer.pad(...)
    
    # DEBUG: Print batch info
    print("\\n=== COLLATED BATCH ===")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}, dtype={v.dtype}")
    print("========================\\n")
    
    return batch
""")

# ============================================================================
# 6. COMMON PATTERNS AND DEBUGGING
# ============================================================================
print_section("6. COMMON PATTERNS & DEBUGGING TIPS")

print("Pattern 1: Label Masking")
print("""
In pre-training, labels are often masked:
  labels[:, ::2, :] = -100  # Mask every other position
  
This means:
  - Only unmasked positions contribute to loss
  - Useful for alternating prediction tasks
  - Check mask ratio: (labels == -100).float().mean()
""")

print("Pattern 2: Attention Mask Construction")
print("""
Causal attention (default):
  attention_mask = torch.tril(torch.ones(seq_len, seq_len))
  
Block attention (for packed graphs):
  Use split_lens to create block-diagonal mask
  Each block is independent (no cross-graph attention)
  
Flex attention:
  Mask created dynamically based on split_lens
  More efficient for variable-length sequences
""")

print("Debugging Checklist:")
print("""
✓ Check tensor shapes match config expectations
✓ Verify no NaN/Inf in inputs: assert not torch.isnan(x).any()
✓ Ensure model and data on same device
✓ Check attention_mask has correct number of 1s per sequence
✓ Verify labels are in valid range (except -100)
✓ Monitor gradient norms during training
✓ Check loss is decreasing (not exploding/vanishing)
""")

print("Common Errors:")
print("""
1. "Expected device cuda but got cpu"
   → Move batch to model device: batch = {k: v.to(device) for k, v in batch.items()}

2. "Shape mismatch at dimension X"
   → Check stacked_feat, batch_size, seq_len match config

3. "CUDA out of memory"
   → Reduce batch_size or max_position_embeddings

4. "Label out of range"
   → Ensure labels < vocab_size (or = -100 for masking)
""")

# ============================================================================
# SUMMARY
# ============================================================================
print_section("SUMMARY")

print("""
PRE-TRAINING FORWARD INPUTS:
  Required: input_ids [B, S, F], attention_mask [B, S], labels [B, S, 1]
  Optional: inputs_raw_embeds, position_ids, split_lens, attn_modes
  
FINE-TUNING FORWARD INPUTS:
  Required: input_ids [B, S, F], attention_mask [B, S], task_labels [B, L]
  Optional: cls_idx, sample_wgt, pretrain_labels, inputs_raw_embeds
  
KEY DIFFERENCES:
  - Pre-training: Predict tokens (labels for MLM)
  - Fine-tuning: Predict graph properties (task_labels for classification)
  - Fine-tuning may include auxiliary pre-training loss
  
DATA FLOW:
  Graph → Tokenization → Stacking → Collation → Model Forward
  
CONFIG IMPACT:
  - pack_tokens: Changes sequence dimensions, adds split_lens
  - flex_attention: Requires split_lens + attn_modes
  - embed_dim: Adds inputs_raw_embeds
  
DEBUGGING:
  - Use hooks or modify collator to inspect inputs
  - Check shapes, dtypes, devices, ranges
  - Monitor label masking ratios
""")

print("\n" + "=" * 80)
print("For actual code execution with real data, run:")
print("  python tests/test_forward_simple.py  (requires dependencies)")
print("  python tests/test_model_forward_inputs.py  (full pipeline)")
print("=" * 80 + "\n")
