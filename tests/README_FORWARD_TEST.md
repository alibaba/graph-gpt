# Model Forward Input Test Scripts

This directory contains test scripts for inspecting what data is passed to the model's `forward()` method during training.

## Available Scripts

### 1. `test_forward_simple.py` (Recommended for quick testing)

A simple, self-contained script that uses dummy data to show model forward inputs.

**Usage:**
```bash
cd /Users/zhaoqifang/Documents/git/graph-gpt
python tests/test_forward_simple.py
```

**What it shows:**
- Pre-training forward pass with dummy tensors
- Fine-tuning forward pass with dummy tensors  
- Real collator batch structure (if dataset is available)

**Output example:**
```
================================================================================
PRE-TRAINING FORWARD PASS TEST
================================================================================

Creating model...

Creating dummy batch:
  batch_size=2, seq_len=64, stacked_feat=1

Input tensors:
input_ids: shape=[2, 64, 1], dtype=torch.int64, min=0.0000, max=511.0000
attention_mask: shape=[2, 64], dtype=torch.float32, min=1.0000, max=1.0000
labels: shape=[2, 64, 1], dtype=torch.int64, min=-100.0000, max=511.0000

Calling model.forward()...

Output:
  head1_loss: 6.245891571044922
  logits: shape=[2, 64, 512], dtype=torch.float32, min=-0.0823, max=0.0891

================================================================================
```

### 2. `test_model_forward_inputs.py` (Advanced, full pipeline testing)

A comprehensive script that runs the actual training pipeline and captures forward inputs.

**Usage:**
```bash
# Test all modes
python tests/test_model_forward_inputs.py --mode all

# Test pre-training only
python tests/test_model_forward_inputs.py --mode pretrain

# Test fine-tuning only
python tests/test_model_forward_inputs.py --mode finetune

# Test with specific dataset
python tests/test_model_forward_inputs.py --mode pretrain --dataset ogbg_molpcba
```

**What it shows:**
- Complete training pipeline setup
- Real data batching through DataLoader
- Hook-based capture of actual forward() calls
- DeepSpeed vs DDP differences (configurable)
- Token packing effects (configurable)

**Features:**
- `ForwardInputInspector` class attaches hook to model.forward()
- Prints tensor shapes, dtypes, devices, min/max values
- Shows both positional and keyword arguments
- Tests collator output structure

## Understanding the Output

### Pre-training Forward Inputs

Typical inputs to `GraphGPTPretrainBase.forward()`:
```python
input_ids: [batch_size, seq_len, stacked_feat]
attention_mask: [batch_size, seq_len]
labels: [batch_size, seq_len, 1] or [batch_size, seq_len, next_n_token]
position_ids: [batch_size, seq_len] (optional)
inputs_raw_embeds: [batch_size, seq_len, embed_dim] (if using node/edge embeddings)
split_lens: List[int] (for flex attention)
attn_modes: List[str] (for flex attention)
```

### Fine-tuning Forward Inputs

Typical inputs to `GraphGPTTaskModel.forward()`:
```python
input_ids: [batch_size, seq_len, stacked_feat]
attention_mask: [batch_size, seq_len]
task_labels: [batch_size, num_labels] or [batch_size, 1]
pretrain_labels: [batch_size, seq_len, 1] (if using auxiliary pre-training)
cls_idx: [batch_size] (for classification pooling)
sample_wgt: [batch_size] (for weighted loss)
position_ids: [batch_size, seq_len] (optional)
inputs_raw_embeds: [batch_size, seq_len, embed_dim] (optional)
split_lens: List[int] (for flex attention)
attn_modes: List[str] (for flex attention)
```

### Key Differences Between Modes

1. **Pre-training:**
   - Focus on token prediction (`labels` for MLM/SMTP)
   - May use generative + discriminative objectives
   - Labels often masked (-100) for some positions

2. **Fine-tuning:**
   - Focus on task prediction (`task_labels`)
   - May include auxiliary pre-training loss
   - Classification/regression targets
   - Sample weights for imbalanced datasets

## Configuration Impact

### Token Packing (`training.pack_tokens`)
When enabled, multiple graphs are packed into one sequence:
- `input_ids` shape changes from `[batch, seq, feat]` to `[1, packed_seq, feat]`
- `split_lens` indicates graph boundaries
- Attention mask becomes block-sparse

### DeepSpeed (`training.use_deepspeed`)
DeepSpeed mode may modify:
- Gradient accumulation behavior
- Mixed precision (FP16/BF16)
- Distributed communication patterns

### Flex Attention
When using flex attention:
- `split_lens`: List of graph lengths in batch
- `attn_modes`: Attention mode per graph ("flash", "memory_efficient", etc.)
- Causal mask is dynamically constructed

## Debugging Tips

1. **Check tensor shapes match config:**
   ```python
   assert input_ids.shape[0] == batch_size
   assert input_ids.shape[1] <= max_position_embeddings
   ```

2. **Verify label masking:**
   ```python
   mask_ratio = (labels == -100).float().mean()
   print(f"Label mask ratio: {mask_ratio:.2%}")
   ```

3. **Inspect attention mask:**
   ```python
   print(f"Attention mask sum: {attention_mask.sum(dim=1)}")  # Sequence lengths
   ```

4. **Check for NaN/Inf:**
   ```python
   assert not torch.isnan(input_ids).any()
   assert not torch.isinf(input_ids).any()
   ```

## Common Issues

### Issue: "Expected device cuda but got cpu"
**Solution:** Ensure model and data are on same device. The test scripts handle this automatically.

### Issue: "Shape mismatch in forward pass"
**Solution:** Check `stacked_feat` dimension matches config. Verify tokenization is correct.

### Issue: "CUDA out of memory"
**Solution:** Reduce `batch_size` or `max_position_embeddings` in test config.

### Issue: "Dataset not found"
**Solution:** Some tests require downloaded datasets. Use dummy data tests first.

## Extending the Tests

To add custom inspections:

```python
def test_custom_config():
    cfg = Config()
    cfg.training.task_type = "pretrain"
    cfg.training.pack_tokens = 0.5  # Enable token packing
    
    # ... setup pipeline ...
    
    # Add your inspection code
    inspector = ForwardInputInspector()
    # ... run training ...
    # Check captured inputs
    for captured in inspector.captured_inputs:
        # Your analysis here
        pass
```

## Related Files

- `src/training/pretrain_mode.py` - Pre-training pipeline
- `src/training/finetune_mode.py` - Fine-tuning pipeline
- `src/models/graphgpt/modeling_pretrain.py` - Pre-training model forward()
- `src/models/graphgpt/modeling_finetune.py` - Fine-tuning model forward()
- `src/data/collator.py` - Batch collation logic
- `src/utils/training_utils.py` - Training step functions
