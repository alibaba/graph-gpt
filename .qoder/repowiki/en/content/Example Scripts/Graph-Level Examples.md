# Graph-Level Examples

<cite>
**Referenced Files in This Document**
- [pcqm4m_v2_pretrain.sh](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh)
- [pcqm4m_v2_supervised.sh](file://examples/graph_lvl/pcqm4m_v2_supervised.sh)
- [molpcba_pretrain.sh](file://examples/graph_lvl/molpcba_pretrain.sh)
- [molpcba_supervised.sh](file://examples/graph_lvl/molpcba_supervised.sh)
- [spice_circuit_pretrain.sh](file://examples/graph_lvl/spice_circuit_pretrain.sh)
- [spice_circuit_supervised.sh](file://examples/graph_lvl/spice_circuit_supervised.sh)
- [pcqm4m-v2.yaml](file://configs/tokenization/graph_lvl/pcqm4m-v2.yaml)
- [ogbg_molpcba.yaml](file://configs/tokenization/graph_lvl/ogbg_molpcba.yaml)
- [spice_circuit.yaml](file://configs/tokenization/graph_lvl/spice_circuit.yaml)
- [train_pretrain.py](file://examples/train_pretrain.py)
- [train_supervised.py](file://examples/train_supervised.py)
- [pipeline.py](file://src/training/pipeline.py)
- [pretrain_mode.py](file://src/training/pretrain_mode.py)
- [finetune_mode.py](file://src/training/finetune_mode.py)
- [base_configs.py](file://src/conf/base_configs.py)
- [misc_utils.py](file://src/utils/misc_utils.py)
</cite>

## Update Summary
**Changes Made**
- Enhanced environment variable system documentation with `env='cpu'|'gpu'` control for distributed training setup
- Added comprehensive coverage of parameter optimization system including `total_tokens`, `warmup_tokens`, and `steps_per_saving`
- Expanded packed token sequence support documentation with `pack_tokens` and `token_per_sample` parameters
- Updated standardized date directory structure documentation with `mid_dir` parameter
- Revised troubleshooting section to include environment variable configuration and distributed training setup

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Environment Variable System](#environment-variable-system)
7. [Parameter Optimization Framework](#parameter-optimization-framework)
8. [Packed Token Sequence Support](#packed-token-sequence-support)
9. [Standardized Directory Structure](#standardized-directory-structure)
10. [Dependency Analysis](#dependency-analysis)
11. [Performance Considerations](#performance-considerations)
12. [Troubleshooting Guide](#troubleshooting-guide)
13. [Conclusion](#conclusion)
14. [Appendices](#appendices)

## Introduction
This document explains the graph-level task examples included in the repository, covering:
- PCQM4M-v2 molecular property prediction (regression)
- MoleculePCBA bioactivity prediction (multi-label classification)
- SPICE circuit analysis (single-label classification)

It details differences between pre-training and fine-tuning configurations, dataset-specific parameters, molecular feature handling, evaluation metrics, script structure, parameter optimization strategies, and computational requirements. Guidance is also provided for adapting these examples to other graph-level datasets, including data preprocessing and performance tuning, with attention to challenges specific to molecular graph learning and circuit analysis tasks.

**Updated** Enhanced with new environment variable system (`env='cpu'|'gpu'`), parameter optimization framework (`total_tokens`, `warmup_tokens`, `steps_per_saving`), packed token sequence support (`pack_tokens`, `token_per_sample`), and standardized date directory structure (`mid_dir`).

## Project Structure
The graph-level examples are organized under examples/graph_lvl with paired pre-training and supervised scripts for each dataset. Tokenization configuration files define dataset semantics, vocabulary, and structure parameters. The training entry points delegate to a unified pipeline that switches behavior via mode strategies for pre-training versus fine-tuning.

```mermaid
graph TB
subgraph "Examples"
PT1["pcqm4m_v2_pretrain.sh"]
SV1["pcqm4m_v2_supervised.sh"]
PT2["molpcba_pretrain.sh"]
SV2["molpcba_supervised.sh"]
PT3["spice_circuit_pretrain.sh"]
SV3["spice_circuit_supervised.sh"]
end
subgraph "Configs"
CFG1["pcqm4m-v2.yaml"]
CFG2["ogbg_molpcba.yaml"]
CFG3["spice_circuit.yaml"]
end
subgraph "Training Entrypoints"
PRETRAIN["train_pretrain.py"]
FINETUNE["train_supervised.py"]
end
subgraph "Pipeline"
PIPE["pipeline.py"]
PMODE["pretrain_mode.py"]
FMODE["finetune_mode.py"]
end
PT1 --> PRETRAIN
SV1 --> FINETUNE
PT2 --> PRETRAIN
SV2 --> FINETUNE
PT3 --> PRETRAIN
SV3 --> FINETUNE
PRETRAIN --> PIPE
FINETUNE --> PIPE
PIPE --> PMODE
PIPE --> FMODE
PT1 -. uses .-> CFG1
SV1 -. uses .-> CFG1
PT2 -. uses .-> CFG2
SV2 -. uses .-> CFG2
PT3 -. uses .-> CFG3
SV3 -. uses .-> CFG3
```

**Diagram sources**
- [pcqm4m_v2_pretrain.sh:1-321](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh#L1-L321)
- [pcqm4m_v2_supervised.sh:1-322](file://examples/graph_lvl/pcqm4m_v2_supervised.sh#L1-L322)
- [molpcba_pretrain.sh:1-321](file://examples/graph_lvl/molpcba_pretrain.sh#L1-L321)
- [molpcba_supervised.sh:1-322](file://examples/graph_lvl/molpcba_supervised.sh#L1-L322)
- [spice_circuit_pretrain.sh:1-321](file://examples/graph_lvl/spice_circuit_pretrain.sh#L1-L321)
- [spice_circuit_supervised.sh:1-322](file://examples/graph_lvl/spice_circuit_supervised.sh#L1-L322)
- [pcqm4m-v2.yaml:1-113](file://configs/tokenization/graph_lvl/pcqm4m-v2.yaml#L1-L113)
- [ogbg_molpcba.yaml:1-115](file://configs/tokenization/graph_lvl/ogbg_molpcba.yaml#L1-L115)
- [spice_circuit.yaml:1-115](file://configs/tokenization/graph_lvl/spice_circuit.yaml#L1-L115)
- [train_pretrain.py:1-19](file://examples/train_pretrain.py#L1-L19)
- [train_supervised.py:1-19](file://examples/train_supervised.py#L1-L19)
- [pipeline.py:1-264](file://src/training/pipeline.py#L1-L264)
- [pretrain_mode.py:1-501](file://src/training/pretrain_mode.py#L1-L501)
- [finetune_mode.py:1-459](file://src/training/finetune_mode.py#L1-L459)

**Section sources**
- [pcqm4m_v2_pretrain.sh:1-321](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh#L1-L321)
- [pcqm4m_v2_supervised.sh:1-322](file://examples/graph_lvl/pcqm4m_v2_supervised.sh#L1-L322)
- [molpcba_pretrain.sh:1-321](file://examples/graph_lvl/molpcba_pretrain.sh#L1-L321)
- [molpcba_supervised.sh:1-322](file://examples/graph_lvl/molpcba_supervised.sh#L1-L322)
- [spice_circuit_pretrain.sh:1-321](file://examples/graph_lvl/spice_circuit_pretrain.sh#L1-L321)
- [spice_circuit_supervised.sh:1-322](file://examples/graph_lvl/spice_circuit_supervised.sh#L1-L322)
- [pcqm4m-v2.yaml:1-113](file://configs/tokenization/graph_lvl/pcqm4m-v2.yaml#L1-L113)
- [ogbg_molpcba.yaml:1-115](file://configs/tokenization/graph_lvl/ogbg_molpcba.yaml#L1-L115)
- [spice_circuit.yaml:1-115](file://configs/tokenization/graph_lvl/spice_circuit.yaml#L1-L115)
- [train_pretrain.py:1-19](file://examples/train_pretrain.py#L1-L19)
- [train_supervised.py:1-19](file://examples/train_supervised.py#L1-L19)
- [pipeline.py:1-264](file://src/training/pipeline.py#L1-L264)
- [pretrain_mode.py:1-501](file://src/training/pretrain_mode.py#L1-L501)
- [finetune_mode.py:1-459](file://src/training/finetune_mode.py#L1-L459)

## Core Components
- Unified training pipeline orchestrating shared setup and delegating to mode-specific strategies.
- PretrainMode for step-level training, token packing, and generation evaluation.
- FinetuneMode for epoch-level training, explicit train/valid/test splits, and evaluation/inference.
- Tokenization configs defining dataset semantics, vocabulary, and structure tokens for molecules and circuits.
- Script families per dataset encapsulating dataset selection, tokenizer config, model sizing, scheduling, and optimization.

Key differences between pre-training and fine-tuning:
- Pre-training focuses on masked-language/modeling objectives with step-based schedules and optional generation evaluation.
- Fine-tuning focuses on explicit task heads (regression/classification) with epoch-based schedules and evaluation cadence.

**Updated** Enhanced with environment variable system for distributed training control, parameter optimization framework for compute budget management, and standardized directory structure for experiment organization.

Key differences between pre-training and fine-tuning:
- Pre-training focuses on masked-language/modeling objectives with step-based schedules and optional generation evaluation.
- Fine-tuning focuses on explicit task heads (regression/classification) with epoch-based schedules and evaluation cadence.

**Section sources**
- [pipeline.py:61-96](file://src/training/pipeline.py#L61-L96)
- [pretrain_mode.py:48-76](file://src/training/pretrain_mode.py#L48-L76)
- [finetune_mode.py:43-71](file://src/training/finetune_mode.py#L43-L71)
- [base_configs.py:132-176](file://src/conf/base_configs.py#L132-L176)

## Architecture Overview
The training pipeline initializes configuration, sets up distributed environments, prepares data and tokenizer, constructs the model, loads checkpoints if applicable, configures optimizers/schedulers, and executes the chosen mode's training loop.

```mermaid
sequenceDiagram
participant SH as "Shell Script"
participant EP as "Entry Point"
participant TP as "TrainingPipeline"
participant PM as "PretrainMode"
participant FM as "FinetuneMode"
SH->>EP : Invoke with tokenization config and overrides
EP->>TP : run()
TP->>TP : _extract_config(), _setup_deepspeed_flag(), _setup_distributed()
TP->>TP : _init_data_configs()
TP->>PM : prepare_data() or FM : prepare_data()
PM-->>TP : tokenizer, dataset, samplers
FM-->>TP : tokenizer, train/valid/test datasets
TP->>TP : _create_model()
TP->>TP : _load_initial_ckp()
TP->>PM : setup_optimizer()/setup_training() or FM : setup_optimizer()/setup_training()
PM->>PM : run_training() with step-level loops
FM->>FM : run_training() with epoch-level loops
TP->>TP : _cleanup()
```

**Diagram sources**
- [train_pretrain.py:12-18](file://examples/train_pretrain.py#L12-L18)
- [train_supervised.py:12-18](file://examples/train_supervised.py#L12-L18)
- [pipeline.py:61-96](file://src/training/pipeline.py#L61-L96)
- [pretrain_mode.py:97-227](file://src/training/pretrain_mode.py#L97-L227)
- [finetune_mode.py:116-200](file://src/training/finetune_mode.py#L116-L200)

## Detailed Component Analysis

### PCQM4M-v2 Molecular Property Prediction
- Dataset and tokenizer: OGB dataset with a 2D molecular representation configuration.
- Pre-training:
  - Objective: masked-language modeling with optional discriminative contrastive learning.
  - Scheduling: total tokens and warmup tokens drive step count; token packing and sampling per saving configured.
  - Evaluation: optional generation evaluation and inference modes.
  - **Updated** Environment control: `env="cpu"` automatically adjusts model size, batch size, and training budget for CPU testing.
  - **Updated** Token packing: When `pack_tokens` is enabled, the script automatically sets `batch_size=1` and calculates `max_position_embeddings=batch_size*token_per_sample` to ensure compatibility with variable-length packed sequences.
- Supervised fine-tuning:
  - Task: graph regression with L1 loss.
  - Scheduling: epochs-based with warmup epochs; evaluation cadence and validation size configurable.
  - Optimization: Adam-style optimizer with EMA enabled by default.

```mermaid
flowchart TD
Start(["Start"]) --> EnvCheck{"env='cpu'?"}
EnvCheck --> |Yes| CPUMode["CPU Testing Mode:<br/>tiny model, reduced batch, limited tokens"]
EnvCheck --> |No| GPUMode["GPU Training Mode:<br/>full model, normal batch"]
CPUMode --> PT["Pretrain PT-MLM<br/>Step-based schedule"]
GPUMode --> PT
PT --> PackCheck{"pack_tokens > 0?"}
PackCheck --> |Yes| ForceBatch["Force batch_size=1<br/>Set max_position_embeddings=batch_size*token_per_sample"]
PackCheck --> |No| NormalCalc["Calculate tokens_per_sample<br/>from dataset statistics"]
ForceBatch --> SV["Supervised Fine-tune<br/>Epoch-based schedule"]
NormalCalc --> SV
SV --> Eval["Evaluate on Valid/Test<br/>Regression metrics"]
Eval --> End(["End"])
```

**Diagram sources**
- [pcqm4m_v2_pretrain.sh:3-102](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh#L3-L102)
- [pcqm4m_v2_pretrain.sh:111-118](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh#L111-L118)
- [pcqm4m_v2_supervised.sh:3-103](file://examples/graph_lvl/pcqm4m_v2_supervised.sh#L3-L103)
- [pcqm4m-v2.yaml:14-25](file://configs/tokenization/graph_lvl/pcqm4m-v2.yaml#L14-L25)

**Section sources**
- [pcqm4m_v2_pretrain.sh:1-321](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh#L1-L321)
- [pcqm4m_v2_supervised.sh:1-322](file://examples/graph_lvl/pcqm4m_v2_supervised.sh#L1-L322)
- [pcqm4m-v2.yaml:1-113](file://configs/tokenization/graph_lvl/pcqm4m-v2.yaml#L1-L113)

### MoleculePCBA Bioactivity Prediction
- Dataset and tokenizer: OGB dataset with molecular semantics and structure tokens.
- Pre-training:
  - Objective: masked-language modeling with configurable weighting and focal gamma.
  - Scheduling: long pre-training budget with substantial total tokens.
  - Evaluation: optional generation and inference modes.
  - **Updated** Environment control: `env="cpu"` automatically adjusts model size, batch size, and training budget for CPU testing.
  - **Updated** Token packing: This dataset currently uses `pack_tokens=1` with `token_per_sample=40` in the pre-training script, enabling packed sequences for variable-length molecules.
- Supervised fine-tuning:
  - Task: multi-label classification with 128 labels.
  - Scheduling: epochs-based with warmup epochs; evaluation cadence and validation size configurable.
  - Optimization: Adam-style optimizer with EMA enabled by default.

```mermaid
flowchart TD
Start2(["Start"]) --> EnvCheck2{"env='cpu'?"}
EnvCheck2 --> |Yes| CPUMode2["CPU Testing Mode:<br/>tiny model, reduced batch, limited tokens"]
EnvCheck2 --> |No| GPUMode2["GPU Training Mode:<br/>mini model, normal batch"]
CPUMode2 --> PT2["Pretrain PT-MLM<br/>Long schedule<br/>pack_tokens=1<br/>token_per_sample=40"]
GPUMode2 --> PT2
PT2 --> SV2["Supervised Fine-tune<br/>Multi-label classification"]
SV2 --> Eval2["Evaluate on Valid/Test<br/>Metrics for multi-label"]
Eval2 --> End2(["End"])
```

**Diagram sources**
- [molpcba_pretrain.sh:3-103](file://examples/graph_lvl/molpcba_pretrain.sh#L3-L103)
- [molpcba_supervised.sh:3-103](file://examples/graph_lvl/molpcba_supervised.sh#L3-L103)
- [ogbg_molpcba.yaml:28-51](file://configs/tokenization/graph_lvl/ogbg_molpcba.yaml#L28-L51)

**Section sources**
- [molpcba_pretrain.sh:1-321](file://examples/graph_lvl/molpcba_pretrain.sh#L1-L321)
- [molpcba_supervised.sh:1-322](file://examples/graph_lvl/molpcba_supervised.sh#L1-L322)
- [ogbg_molpcba.yaml:1-115](file://configs/tokenization/graph_lvl/ogbg_molpcba.yaml#L1-L115)

### SPICE Circuit Analysis
- Dataset and tokenizer: Custom dataset with circuit semantics and structure tokens.
- Pre-training:
  - Objective: masked-language modeling with configurable generation settings.
  - Scheduling: moderate pre-training budget; token packing and sampling per saving configured.
  - Evaluation: optional generation and inference modes.
  - **Updated** Environment control: `env="cpu"` automatically adjusts model size, batch size, and training budget for CPU testing.
  - **Updated** Token packing: This dataset currently uses `pack_tokens=1` with `token_per_sample=50` in the pre-training script, enabling packed sequences for variable-length circuits.
- Supervised fine-tuning:
  - Task: single-label classification with 14 classes.
  - Scheduling: epochs-based with warmup epochs; evaluation cadence and validation size configurable.
  - Optimization: Adam-style optimizer with EMA disabled by default.

```mermaid
flowchart TD
Start3(["Start"]) --> EnvCheck3{"env='cpu'?"}
EnvCheck3 --> |Yes| CPUMode3["CPU Testing Mode:<br/>tiny model, reduced batch, limited tokens"]
EnvCheck3 --> |No| GPUMode3["GPU Training Mode:<br/>tiny model, normal batch"]
CPUMode3 --> PT3["Pretrain PT-MLM<br/>Moderate schedule<br/>pack_tokens=1<br/>token_per_sample=50"]
GPUMode3 --> PT3
PT3 --> SV3["Supervised Fine-tune<br/>Single-label classification"]
SV3 --> Eval3["Evaluate on Valid/Test<br/>Metrics for single-label"]
Eval3 --> End3(["End"])
```

**Diagram sources**
- [spice_circuit_pretrain.sh:3-103](file://examples/graph_lvl/spice_circuit_pretrain.sh#L3-L103)
- [spice_circuit_supervised.sh:3-103](file://examples/graph_lvl/spice_circuit_supervised.sh#L3-L103)
- [spice_circuit.yaml:28-51](file://configs/tokenization/graph_lvl/spice_circuit.yaml#L28-L51)

**Section sources**
- [spice_circuit_pretrain.sh:1-321](file://examples/graph_lvl/spice_circuit_pretrain.sh#L1-L321)
- [spice_circuit_supervised.sh:1-322](file://examples/graph_lvl/spice_circuit_supervised.sh#L1-L322)
- [spice_circuit.yaml:1-115](file://configs/tokenization/graph_lvl/spice_circuit.yaml#L1-L115)

### Tokenization and Data Semantics
- Tokenization configs define:
  - Semantic attributes for nodes and edges.
  - Structure tokens (BOS/EOS/new-node, edge tokens, graph summary).
  - Vocabulary and reserved tokens.
  - Task conversion hooks for pre-training objectives.
- Differences across datasets:
  - PCQM4M-v2: molecular semantics with node and edge dimensions suitable for regression.
  - MoleculePCBA: molecular semantics with multi-label classification targets.
  - SPICE: circuit semantics with fewer node dimensions and single-label classification.

**Section sources**
- [pcqm4m-v2.yaml:26-113](file://configs/tokenization/graph_lvl/pcqm4m-v2.yaml#L26-L113)
- [ogbg_molpcba.yaml:28-115](file://configs/tokenization/graph_lvl/ogbg_molpcba.yaml#L28-L115)
- [spice_circuit.yaml:28-115](file://configs/tokenization/graph_lvl/spice_circuit.yaml#L28-L115)

### Pre-Training vs Fine-Tuning Configuration Differences
- Pre-training:
  - Step-based schedule driven by total tokens and warmup tokens.
  - Optional token packing and generation evaluation.
  - Discriminative contrastive learning option.
  - **Updated** Environment variable system: `env='cpu'|'gpu'` controls distributed training setup and resource allocation.
  - **Updated** Token packing mechanism: When `pack_tokens > 0`, the system uses a fixed `tokens_per_sample` equal to `max_position_embeddings` and forces `batch_size=1` for compatibility with variable-length packed sequences.
- Fine-tuning:
  - Epoch-based schedule with warmup epochs.
  - Explicit train/valid/test loaders and evaluation cadence.
  - Task-specific head configuration (regression or multi/single-label classification).

**Section sources**
- [pretrain_mode.py:50-76](file://src/training/pretrain_mode.py#L50-L76)
- [finetune_mode.py:43-71](file://src/training/finetune_mode.py#L43-L71)
- [base_configs.py:54-73](file://src/conf/base_configs.py#L54-L73)
- [base_configs.py:166-176](file://src/conf/base_configs.py#L166-L176)

## Environment Variable System
The `env` environment variable controls distributed training setup and resource allocation across all graph-level scripts.

### CPU Testing Mode (`env="cpu"`)
When `env="cpu"` is set, scripts automatically adjust configurations for CPU-only testing:
- Reduces model size to `"tiny"` architecture
- Decreases batch size to accommodate CPU memory constraints
- Limits total training tokens and warmup tokens to reasonable CPU testing amounts
- Disables DeepSpeed configuration for native DDP
- Reduces validation and evaluation overhead

### GPU Training Mode (`env="gpu"`)
When `env="gpu"` is set, scripts use full GPU training configurations:
- Enables DeepSpeed distributed training
- Uses appropriate model sizes for GPU resources
- Maintains full training budgets and schedules
- Supports multi-GPU distributed training

**Section sources**
- [pcqm4m_v2_pretrain.sh:3-102](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh#L3-L102)
- [molpcba_pretrain.sh:3-103](file://examples/graph_lvl/molpcba_pretrain.sh#L3-L103)
- [spice_circuit_pretrain.sh:3-103](file://examples/graph_lvl/spice_circuit_pretrain.sh#L3-L103)
- [pcqm4m_v2_supervised.sh:3-103](file://examples/graph_lvl/pcqm4m_v2_supervised.sh#L3-L103)
- [molpcba_supervised.sh:3-103](file://examples/graph_lvl/molpcba_supervised.sh#L3-L103)
- [spice_circuit_supervised.sh:3-103](file://examples/graph_lvl/spice_circuit_supervised.sh#L3-L103)

## Parameter Optimization Framework
The parameter optimization system manages compute budgets and training schedules through three key parameters:

### Total Tokens (`total_tokens`)
Defines the overall compute budget for training:
- Pre-training: Controls total number of tokens processed across entire training
- Fine-tuning: Converted to equivalent epoch-based schedule using tokens_per_sample
- Default values: 1e9 for pre-training, adjusted based on dataset complexity

### Warmup Tokens (`warmup_tokens`)
Controls the initial warmup period:
- Defines number of tokens used for learning rate warmup
- Ensures stable training initialization
- Typically 10% of total_tokens for balanced warmup

### Steps Per Saving (`steps_per_saving`)
Controls checkpoint frequency:
- Determines how often model checkpoints are saved
- Balances training progress preservation with storage requirements
- Default: 1000 steps for pre-training, optimized for long-running jobs

**Section sources**
- [base_configs.py:42-51](file://src/conf/base_configs.py#L42-L51)
- [base_configs.py:53-65](file://src/conf/base_configs.py#L53-L65)
- [pcqm4m_v2_pretrain.sh:38-41](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh#L38-L41)
- [molpcba_pretrain.sh:38-41](file://examples/graph_lvl/molpcba_pretrain.sh#L38-L41)
- [spice_circuit_pretrain.sh:38-41](file://examples/graph_lvl/spice_circuit_pretrain.sh#L38-L41)

## Packed Token Sequence Support
The packed token sequence system enables efficient processing of variable-length graphs through token packing mechanisms.

### Token Packing Configuration
- `pack_tokens`: Enable/disable token packing (1 to enable, 0 to disable)
- `token_per_sample`: Maximum tokens allowed per sample in packed sequences
- Automatic batch size adjustment: When pack_tokens > 0, batch_size is forced to 1

### Token Packing Mechanism
When `pack_tokens > 0`:
1. Forces `batch_size = 1` for compatibility with variable-length packed sequences
2. Calculates `max_position_embeddings = batch_size × token_per_sample`
3. Uses fixed `tokens_per_sample = max_position_embeddings` instead of data-derived estimates
4. Enables efficient packing of multiple graphs into single sequences

### Benefits and Considerations
- **Benefits**: Reduces padding overhead, improves memory utilization, handles variable-length sequences
- **Considerations**: Single batch processing limits throughput, requires careful token_per_sample sizing

**Section sources**
- [pcqm4m_v2_pretrain.sh:29-31](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh#L29-L31)
- [pcqm4m_v2_pretrain.sh:111-118](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh#L111-L118)
- [molpcba_pretrain.sh:29-31](file://examples/graph_lvl/molpcba_pretrain.sh#L29-L31)
- [spice_circuit_pretrain.sh:29-31](file://examples/graph_lvl/spice_circuit_pretrain.sh#L29-L31)
- [pretrain_mode.py:222-233](file://src/training/pretrain_mode.py#L222-L233)

## Standardized Directory Structure
The `mid_dir` parameter provides standardized date-based directory organization for experiments.

### Directory Structure Pattern
```
{dataset_prefix}/{mid_dir}{experiment_details}
```

Where:
- `{dataset_prefix}`: Dataset identifier (e.g., "pcqm4m-v2", "ogbg_molpcba", "spice_circuit")
- `{mid_dir}`: Date-based directory (e.g., "202603/")
- `{experiment_details}`: Experiment-specific suffix with hyperparameters

### Example Directory Organization
```
exp/models/
├── pcqm4m-v2/
│   └── 202603/
│       └── pt_h768_l12_tk1e9_b64_mpe1024...
├── ogbg_molpcba/
│   └── 202603/
│       └── pt_h256_l4_tk1e9_b256_mpe1024...
└── spice_circuit/
    └── 202603/
        └── pt_h128_l2_tk1e9_b128_mpe1024...
```

**Section sources**
- [pcqm4m_v2_pretrain.sh:50](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh#L50)
- [molpcba_pretrain.sh:50](file://examples/graph_lvl/molpcba_pretrain.sh#L50)
- [spice_circuit_pretrain.sh:50](file://examples/graph_lvl/spice_circuit_pretrain.sh#L50)
- [pcqm4m_v2_supervised.sh:54](file://examples/graph_lvl/pcqm4m_v2_supervised.sh#L54)
- [molpcba_supervised.sh:54](file://examples/graph_lvl/molpcba_supervised.sh#L54)
- [spice_circuit_supervised.sh:54](file://examples/graph_lvl/spice_circuit_supervised.sh#L54)

## Dependency Analysis
The training pipeline coordinates shared setup and delegates to mode-specific strategies. The mode classes depend on data utilities, collators, and model factories. Tokenization configs feed into model configuration initialization and tokenizer construction.

```mermaid
graph LR
TP["TrainingPipeline"] --> PM["PretrainMode"]
TP --> FM["FinetuneMode"]
PM --> Coll["DataCollatorForGST"]
FM --> Coll
PM --> DS["DeepSpeed/Native Optimizer"]
FM --> DS
TP --> CFG["Config (Tokenization/Model/Training)"]
CFG --> TK["Tokenizer"]
CFG --> MC["Model Config"]
```

**Diagram sources**
- [pipeline.py:15-96](file://src/training/pipeline.py#L15-L96)
- [pretrain_mode.py:24-45](file://src/training/pretrain_mode.py#L24-L45)
- [finetune_mode.py:12-38](file://src/training/finetune_mode.py#L12-L38)
- [base_configs.py:186-204](file://src/conf/base_configs.py#L186-L204)

**Section sources**
- [pipeline.py:1-264](file://src/training/pipeline.py#L1-L264)
- [pretrain_mode.py:1-501](file://src/training/pretrain_mode.py#L1-L501)
- [finetune_mode.py:1-459](file://src/training/finetune_mode.py#L1-L459)
- [base_configs.py:1-368](file://src/conf/base_configs.py#L1-L368)

## Performance Considerations
- Pre-training:
  - Use DeepSpeed for large-scale step-based training; configure steps per saving and logging frequency to balance throughput and checkpoint overhead.
  - Token packing can reduce padding overhead but requires careful estimation of tokens per sample.
  - Adjust batch size and world size to meet target total tokens efficiently.
  - **Updated** Environment variable optimization: Use `env="cpu"` for development and debugging, `env="gpu"` for production training.
  - **Updated** Token packing considerations: When using token packing (`pack_tokens > 0`), the system automatically sets `batch_size=1` and uses `tokens_per_sample=max_position_embeddings`. This ensures compatibility with variable-length packed sequences but limits throughput.
- Fine-tuning:
  - Epoch-based training benefits from warmup epochs proportional to total epochs.
  - Validation size and evaluation cadence should balance accuracy monitoring with training time.
  - EMA can improve generalization at a modest memory cost.
- Hardware:
  - Multi-GPU with DeepSpeed is recommended for pre-training; CPU testing modes are available for quick checks.
  - Ensure sufficient memory for model sizes and batch configurations; gradient checkpointing is enabled in model creation.

**Section sources**
- [pcqm4m_v2_pretrain.sh:111-118](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh#L111-L118)
- [pretrain_mode.py:169-197](file://src/training/pretrain_mode.py#L169-L197)
- [misc_utils.py:507-539](file://src/utils/misc_utils.py#L507-L539)

## Troubleshooting Guide
- Resuming training:
  - The pipeline checks for an existing log file and prefers resuming from the current output directory rather than loading a pretrain checkpoint path.
- Checkpoint loading:
  - When resuming, DeepSpeed and native DDP have distinct loading paths; ensure the correct checkpoint format is present.
- Distributed training:
  - Verify DeepSpeed configuration file path and environment variables; ensure ranks and world size match hardware setup.
  - **Updated** Environment variable issues: Ensure `env='cpu'|'gpu'` is properly set; CPU mode disables DeepSpeed automatically.
- Evaluation-only or inference-only modes:
  - Pre-training supports evaluation-only and inference-only modes; fine-tuning supports evaluation-only and inference-only modes with epoch-based iteration.
- **Updated** Token packing issues:
  - When `pack_tokens > 0`, the system automatically forces `batch_size=1` and calculates `max_position_embeddings=batch_size*token_per_sample`.
  - If you encounter unexpected batch size behavior, check the `pack_tokens` parameter in your script.
  - The `token_per_sample` parameter controls the maximum number of tokens allowed per sample in packed sequences.
- **Updated** Directory structure issues:
  - Ensure `mid_dir` follows the standardized date format (YYYYMM); incorrect format may cause directory creation failures.

**Section sources**
- [pipeline.py:129-136](file://src/training/pipeline.py#L129-L136)
- [pipeline.py:181-202](file://src/training/pipeline.py#L181-L202)
- [pretrain_mode.py:240-265](file://src/training/pretrain_mode.py#L240-L265)
- [finetune_mode.py:351-358](file://src/training/finetune_mode.py#L351-L358)
- [pcqm4m_v2_pretrain.sh:111-118](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh#L111-L118)

## Conclusion
The repository provides robust, reusable examples for graph-level tasks across molecular and circuit domains. Pre-training establishes strong generative representations with step-based schedules, while fine-tuning adapts models to specific tasks with epoch-based schedules and explicit evaluation. Tokenization configs capture dataset-specific semantics and structure, enabling consistent model configuration across datasets. The unified pipeline and mode strategies simplify adaptation to new graph-level datasets with minimal code changes.

**Updated** The enhanced environment variable system (`env='cpu'|'gpu'`), parameter optimization framework (`total_tokens`, `warmup_tokens`, `steps_per_saving`), packed token sequence support (`pack_tokens`, `token_per_sample`), and standardized directory structure (`mid_dir`) provide comprehensive control over training workflows, making the system adaptable to various computational environments and dataset characteristics.

## Appendices

### Adapting to Other Graph-Level Datasets
- Data preprocessing:
  - Define a new tokenization YAML under configs/tokenization/graph_lvl with dataset semantics, node/edge dimensions, and structure tokens.
  - Ensure vocabulary and reserved tokens align with the dataset's attribute ranges.
- Script structure:
  - Create paired pretrain and supervised shell scripts mirroring the dataset-specific examples.
  - Set dataset source/name, tokenizer class, and tokenization config path.
  - Tune scheduling, batch size, and optimizer hyperparameters per dataset scale and task type.
  - **Updated** Include environment variable configuration (`env='cpu'|'gpu'`) for flexible deployment.
- Performance tuning:
  - For pre-training, adjust total tokens and warmup tokens to reach desired compute budgets; consider token packing and steps per saving.
  - For fine-tuning, adjust epochs and warmup epochs; monitor validation metrics and tune learning rate and weight decay accordingly.
- Challenges:
  - Molecular graphs: handle diverse atom/bond types and varying sizes; consider multi-label classification targets and regression scaling.
  - Circuit graphs: leverage structural tokens and fewer node attributes; ensure class imbalance is addressed in single-label classification.
- **Updated** Advanced configuration considerations:
  - Use `pack_tokens > 0` to enable token packing for variable-length sequences.
  - Set `token_per_sample` appropriately based on expected sequence lengths in your dataset.
  - Be aware that token packing forces `batch_size=1` for compatibility with variable-length packed sequences.
  - Configure `mid_dir` with standardized date format for organized experiment tracking.

### Understanding the `token_per_sample` Parameter
- Purpose: Controls the maximum number of tokens allowed per sample when using token packing.
- Behavior: When `pack_tokens > 0`, the system:
  - Forces `batch_size=1` for compatibility with variable-length packed sequences
  - Sets `max_position_embeddings=batch_size*token_per_sample`
  - Uses a fixed `tokens_per_sample=max_position_embeddings` instead of estimating from data
- Usage guidelines:
  - Set `token_per_sample` based on your dataset's typical sequence lengths
  - For datasets with highly variable graph sizes, choose a value that accommodates most samples
  - Monitor memory usage as larger `token_per_sample` values increase memory requirements

### Environment Variable Configuration Reference
- `env="cpu"`: Activates CPU testing mode with reduced model size, batch size, and training budget
- `env="gpu"`: Activates GPU training mode with full model configurations and distributed training
- Automatic adjustments: Model size, batch size, token counts, and DeepSpeed configuration based on environment setting

**Section sources**
- [pcqm4m_v2_pretrain.sh:29-31](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh#L29-L31)
- [pcqm4m_v2_pretrain.sh:111-118](file://examples/graph_lvl/pcqm4m_v2_pretrain.sh#L111-L118)
- [pretrain_mode.py:169-197](file://src/training/pretrain_mode.py#L169-L197)
- [misc_utils.py:507-539](file://src/utils/misc_utils.py#L507-L539)
