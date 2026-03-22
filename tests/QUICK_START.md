# Forward Input Test Scripts - Quick Start

## 快速开始 (Quick Start)

### 最简单的测试（无需依赖）

```bash
cd /Users/zhaoqifang/Documents/git/graph-gpt
python tests/test_forward_minimal.py
```

这个脚本会打印详细的文档，说明不同训练配置下 model.forward() 的输入。

### 输出示例

```
================================================================================
  PRE-TRAINING MODE (GraphGPTPretrainBase)
================================================================================

Forward Method Signature:
def forward(
    input_ids: [batch_size, seq_len, stacked_feat],
    attention_mask: [batch_size, seq_len],
    labels: [batch_size, seq_len, 1],
    ...
) -> DoubleHeadsModelOutput

input_ids:
  Shape: [batch_size, seq_len, stacked_feat]
  Dtype: int64
  Purpose: Token IDs for each position

attention_mask:
  Shape: [batch_size, seq_len]
  Dtype: float32
  Purpose: 1 for real tokens, 0 for padding

labels:
  Shape: [batch_size, seq_len, 1]
  Dtype: int64
  Purpose: Target tokens for prediction (-100 = ignore)
```

## 三个测试脚本对比

| 脚本 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| `test_forward_minimal.py` | ✅ 无需依赖<br>✅ 详细文档<br>✅ 即刻运行 | ❌ 不实际执行 | 快速了解结构 |
| `test_forward_simple.py` | ✅ 实际执行<br>✅ 虚拟数据<br>✅ 真实输出 | ❌ 需安装依赖 | 测试代码逻辑 |
| `test_model_forward_inputs.py` | ✅ 完整 pipeline<br>✅ 真实数据<br>✅ Hook 捕获 | ❌ 依赖复杂<br>❌ 运行慢 | 深度调试 |

## Pre-training vs Fine-tuning 输入对比

### Pre-training (预训练)

```python
# GraphGPTPretrainBase.forward()
input_ids = torch.randint(0, vocab_size, [2, 64, 1])      # token IDs
attention_mask = torch.ones(2, 64)                        # 掩码
labels = torch.randint(0, vocab_size, [2, 64, 1])         # 预测目标
labels[:, ::2, :] = -100                                  # 每隔一个 mask

output = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
)
# 输出：DoubleHeadsModelOutput(loss=logits)
```

### Fine-tuning (微调)

```python
# GraphGPTTaskModel.forward()
input_ids = torch.randint(0, vocab_size, [2, 64, 1])      # token IDs
attention_mask = torch.ones(2, 64)                        # 掩码
task_labels = torch.randn(2, 1)                           # 回归目标

output = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    task_labels=task_labels,
)
# 输出：DoubleHeadsModelOutput(task_loss, task_logits)
```

## 关键区别

| 特性 | Pre-training | Fine-tuning |
|------|--------------|-------------|
| **必需输入** | input_ids, attention_mask, labels | input_ids, attention_mask, task_labels |
| **标签类型** | Token-level (序列标注) | Graph-level (分类/回归) |
| **损失计算** | 预测下一个 token | 预测图属性 |
| **可选辅助** | SMTP, MLM | Denoising, 辅助预训练 |

## 配置影响

### 1. Token Packing (`pack_tokens=0.5`)

**Without Packing:**
```
input_ids: [batch_size=32, seq_len=128, feat=1]
每个序列是一个独立的图
```

**With Packing:**
```
input_ids: [batch_size=1, seq_len=4096, feat=1]
多个图打包成一个长序列
split_lens: [128, 256, 96, ...]  # 指示每个图的长度
```

### 2. Flex Attention

添加特殊输入:
```python
split_lens = [64, 128, 96, 32]           # 图边界
attn_modes = ['flash', 'sdpa', 'flex']   # 注意力模式
```

效果：创建块对角注意力掩码，防止跨图注意。

### 3. Node/Edge Embeddings

当 `model.embed_dim > 0`:
```python
inputs_raw_embeds: [batch_size, seq_len, embed_dim=128]
# 原始节点/边特征，投影后加到 token embeddings 上
```

## 如何在训练中打印输入

### 方法 1: 修改 training_utils.py

找到 `batch_training` 或 `ft_batch_training` 函数，在开头添加:

```python
def batch_training(data, model, train_cfg, train_stats, opt_stats):
    # ========== 添加调试输出 ==========
    print(f"\n[DEBUG] Batch {train_stats.j}")
    print("Input tensors:")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                  f"device={value.device}")
            if value.numel() > 0:
                print(f"        min={value.min().item():.4f}, max={value.max().item():.4f}")
    print("=" * 60 + "\n")
    # ===================================
    
    # 原有训练代码...
    output = model(**data)
    ...
```

### 方法 2: 使用 Hook (推荐)

在模型创建后添加:

```python
class ForwardHook:
    def __init__(self):
        self.inputs = []
    
    def __call__(self, module, args, kwargs):
        # 只打印 kwargs
        print("\n=== FORWARD CALLED ===")
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape} on {v.device}")
        print("=" * 22 + "\n")

# 在 setup_training 中
hook = ForwardHook()
model.register_forward_hook(hook)
```

### 方法 3: 修改 Collator

在 `src/data/collator.py` 的 `__call__` 方法末尾添加:

```python
def __call__(self, graphs, return_tensors=None):
    features = [...]
    batch = self.tokenizer.pad(...)
    
    # ========== 调试输出 ==========
    print("\n[COLLATOR OUTPUT]")
    for key, tensor in batch.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {key}: {tensor.shape}, dtype={tensor.dtype}")
    print("=" * 30 + "\n")
    # ==============================
    
    return batch
```

## 常见调试场景

### 场景 1: 检查数据是否正确加载

```bash
# 运行 minimal 测试查看理论结构
python tests/test_forward_minimal.py | grep -A 5 "PRE-TRAINING MODE"
```

### 场景 2: 验证形状匹配

在实际训练中:
```python
assert input_ids.shape[0] == batch_size, f"Expected batch={batch_size}, got {input_ids.shape[0]}"
assert input_ids.shape[1] <= max_position_embeddings, f"Sequence too long: {input_ids.shape[1]} > {max_position_embeddings}"
```

### 场景 3: 检查 Label 掩码

```python
mask_ratio = (labels == -100).float().mean().item()
print(f"Label mask ratio: {mask_ratio:.2%}")
# 正常范围：30%-70% (取决于任务)
```

### 场景 4: 监控梯度

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 1.0:
            print(f"Large gradient in {name}: {grad_norm:.4f}")
```

## 故障排除

### 问题 1: ModuleNotFoundError

**症状:** `No module named 'timm'`

**解决:** 
```bash
# 使用 minimal 版本（无需依赖）
python tests/test_forward_minimal.py
```

### 问题 2: CUDA Out of Memory

**症状:** `RuntimeError: CUDA out of memory`

**解决:**
```yaml
# 在 config.yaml 中减小 batch_size
training:
  batch_size: 8  # 从 128 改为 8
  max_length: 256  # 从 1024 改为 256
```

### 问题 3: 设备不匹配

**症状:** `Expected all tensors to be on the same device`

**解决:**
```python
# 确保所有张量在同一设备
device = model.device
data = {k: v.to(device) for k, v in data.items()}
```

## 相关文档

- 📄 `tests/README_FORWARD_TEST.md` - 详细英文文档
- 📄 `tests/测试脚本说明.md` - 详细中文文档
- 📄 `tests/test_forward_minimal.py` - 免依赖测试脚本
- 📄 `src/training/` - 训练流程源码
- 📄 `src/models/graphgpt/` - 模型定义源码

## 总结

**三步理解 Forward 输入:**

1. **理论学习** → 运行 `test_forward_minimal.py` 查看详细文档
2. **实际验证** → 在训练代码中添加调试输出
3. **深入调试** → 使用 hook 或修改 collator

**核心要点:**
- Pre-training: `labels` 用于 token 预测
- Fine-tuning: `task_labels` 用于图属性预测
- 配置会影响输入形状（packing, flex attention, embeddings）
- 调试时重点检查：形状、dtype、device、值范围
