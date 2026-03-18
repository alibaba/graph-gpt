# Changelog

All notable changes to this project will be documented in this file.

## [0.7.0] - 2026-03-18

### Code Refactoring

Major codebase refactoring across three areas to improve modularity, reduce duplication,
and make the project easier to extend.

#### Proposal 1: Model Decomposition

Decomposed the monolithic `modeling_graphgpt.py` into focused, single-responsibility modules:
- `modeling_common.py`: Shared infrastructure (output dataclasses, initialization helpers, stacked feature aggregation)
- `modeling_helpers.py`: Standalone helper functions organized by category (attention masks, embeddings, loss functions, label/logit preparation, SMTP handling, 3D position transforms)
- `modeling_pretrain.py`: Pre-training models (`GraphGPTPretrainBase`, `GraphGPTPosPred`)
- `modeling_finetune.py`: Fine-tuning models (`GraphGPTTaskModel`, `GraphGPTDoubleHeadsModel`, `GraphGPTDenoisingRegressionDoubleHeadsModel`)
- `configuration_graphgpt.py`: Externalized `GraphGPTConfig` with `convert_to_legacy_config()` bridge function
- `modeling_graphgpt.py` retained as a thin backward-compatible re-export shim (29 lines)

#### Proposal 2: Data Source Generalization

Replaced the monolithic `data_sources.py` (2,036 lines) with a registry-driven factory pattern:
- `_graph_factory.py`: Generic `DatasetSpec` dataclass and `read_graph_dataset()` factory function
- `_readers/pcqm4mv2.py`: Specialized reader for PCQM4M-v2 molecular datasets
- `_readers/edge_level.py`: Readers for edge-level prediction datasets (link prediction, edge classification)
- `_readers/node_level.py`: Readers for node-level prediction datasets (OGB, TUDataset)
- `_helpers/edge_formatting.py`, `graph_utils.py`, `node_encoding.py`: Extracted reusable utilities
- `data_sources.py` reduced to 412 lines (80% reduction), serving as a streamlined entry point with dataset spec registrations

Adding new datasets now requires only defining a `DatasetSpec`; no changes to reader logic needed.

#### Proposal 3: Unified Training Pipeline

Consolidated duplicated training logic between `train_pretrain.py` (458 lines) and `train_supervised.py` (412 lines) using a strategy pattern:
- `src/training/pipeline.py`: `TrainingPipeline` class with 9 shared methods orchestrating a 17-step training flow (config extraction, distributed setup, model creation, checkpoint handling, cleanup)
- `src/training/mode.py`: `TrainingMode` abstract base class defining the strategy interface (6 abstract methods, 3 overridable properties)
- `src/training/pretrain_mode.py`: `PretrainMode` strategy (step-level checkpointing, token packing, generation evaluation)
- `src/training/finetune_mode.py`: `FinetuneMode` strategy (epoch-level evaluation, layer freezing, eval_only/infer_only modes)
- `examples/train_pretrain.py` and `examples/train_supervised.py` reduced to ~18 line thin wrappers (96% reduction)

### Configuration

- Add externalized YAML configuration for position pre-training and denoising regression parameters (`configs/model/base.yaml`)
- Add structured dataclass-based model configuration (`src/conf/model/model_configs.py`)

## [0.6.1] - 2025-12-23

### Model
- Add contrastive loss pre-training for graph-level datasets like PCQM4M-v2
- Add inferring graph-level embeddings with model trained with contrastive loss

### Code Refactoring
- Code refactoring for edge-level example `ogbl-ppa`

## [0.6.0] - 2025-11-19

### Model
- Add generation functionality analogous to discrete diffusion LM after pre-trained with `pretrain-mlm` objective

### Code Refactoring
- Manage configurations using `omegaconf`, `hydra` and yaml files.

### Dependencies upgrade
- Upgrade `python` to 3.10
- Upgrade `pytorch` to 2.5.1
- Upgrade `transformers` to 4.53.3

## [0.5.0] - 2025-05-15

### Model
- Refactor model architectures
- Release 4 checkpoints for the PCQM4M-v2 dataset in [ModelScope](https://www.modelscope.cn/organization/Alibaba-DT) 

### Other
- Code refactoring.
- Update paper.

## [0.4.0] - 2024-10-13

### Dataset
- Add edge-level example `ogbl-citation2` and `ogbl-wikikg2`
- Add node-level example `ogbn-products`

### Other
- Code refactoring.
- Update README to include details of Eulerian sequence

## [0.3.1] - 2024-08-18

### Model
- Add drop path to regularize large models, and it works quite well for deep models

### Other
- Add one package dependency: `timm`, to implement EMA
- Update README to include details of Eulerian sequence and cyclic node re-index.
- Code refactoring.
- Tokenization config json refactoring.
- Update vocab by adding some special tokens, e.g., `<bos>`, `<new>`, `<mask>` and etc.
- Turn off optimizer offload in deepspeed config to boost the training speed.

## [0.3.0] - 2024-07-09

### Dataset
- Add toy examples using dataset `TUDataset/reddit_threads`.
- Add `src/utils/dataset_utils.py::StructureDataset` to generate random non-attributed graphs.
  for structural understanding pre-training

#### Dataset processing: tokenization
- Add a new type of node re-indexing: `cyclic`.

### Model
- Add two new backbones, i.e., backbones from Google's [Bert](https://arxiv.org/abs/1810.04805)
  and OpenAI's [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

### Optimization

#### Optimization Objective
- Add a new pre-train objective, i.e., `Masked Language Modeling` from [Bert](https://arxiv.org/abs/1810.04805).

### Other
- Code refactoring.
- Upgrade two package dependencies: `deepspeed==0.14.0` and `transformers==4.38.2`.
- Add one package dependency: `tensorboardX`.
