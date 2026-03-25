# Vocabulary Building System

<cite>
**Referenced Files in This Document**
- [vocab_builder.py](file://src/data/vocab_builder.py)
- [tokenizer.py](file://src/data/tokenizer.py)
- [tokenizer_utils.py](file://src/utils/tokenizer_utils.py)
- [base.yaml](file://configs/tokenization/base.yaml)
- [token_configs.py](file://src/conf/tokenization/token_configs.py)
- [mol_utils.py](file://src/utils/mol_utils.py)
- [pretrain_mode.py](file://src/training/pretrain_mode.py)
- [nx_utils.py](file://src/utils/nx_utils.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)
10. [Appendices](#appendices)

## Introduction
This document explains the vocabulary building system used to construct and manage token vocabularies for graph tokenization. It covers how vocabularies are assembled from structure tokens, semantic tokens, and special tokens; how token frequency influences vocabulary design; and how vocabulary size impacts memory and model efficiency. It also documents configuration options for file formats, pruning strategies, and dynamic updates, along with practical examples from the codebase.

## Project Structure
The vocabulary system spans three primary areas:
- Vocabulary construction and persistence: building, saving, and loading vocabularies
- Tokenization pipeline: mapping tokens to IDs and preparing inputs for training
- Configuration: specifying structure and semantics vocabularies, reserved tokens, and special tokens

```mermaid
graph TB
subgraph "Vocabulary Construction"
VB["vocab_builder.py<br/>build_vocab, get_structure_vocab,<br/>get_semantics_vocab, save/load"]
end
subgraph "Tokenization Pipeline"
TK["tokenizer.py<br/>GSTTokenizer, load_vocab,<br/>convert_tokens_to_ids, prepare_inputs_for_task"]
TU["tokenizer_utils.py<br/>prepare_inputs_for_task_*"]
end
subgraph "Configuration"
CFG["base.yaml<br/>tokenization config"]
TCFG["token_configs.py<br/>TokenizationConfig, Structure/Smantics sub-configs"]
end
subgraph "Utilities"
MU["mol_utils.py<br/>read_complete_*_features_ds"]
NXU["nx_utils.py<br/>understand_structure, *_nx functions"]
end
subgraph "Training Integration"
PM["pretrain_mode.py<br/>build_vocab, instantiate tokenizer"]
end
CFG --> VB
TCFG --> VB
VB --> TK
TK --> TU
MU --> VB
NXU --> TK
PM --> VB
PM --> TK
```

**Diagram sources**
- [vocab_builder.py:188-218](file://src/data/vocab_builder.py#L188-L218)
- [tokenizer.py:30-120](file://src/data/tokenizer.py#L30-L120)
- [tokenizer_utils.py:222-363](file://src/utils/tokenizer_utils.py#L222-L363)
- [base.yaml:22-117](file://configs/tokenization/base.yaml#L22-L117)
- [token_configs.py:115-127](file://src/conf/tokenization/token_configs.py#L115-L127)
- [mol_utils.py:39-52](file://src/utils/mol_utils.py#L39-L52)
- [nx_utils.py:17-50](file://src/utils/nx_utils.py#L17-L50)
- [pretrain_mode.py:154-166](file://src/training/pretrain_mode.py#L154-L166)

**Section sources**
- [vocab_builder.py:188-218](file://src/data/vocab_builder.py#L188-L218)
- [tokenizer.py:30-120](file://src/data/tokenizer.py#L30-L120)
- [base.yaml:22-117](file://configs/tokenization/base.yaml#L22-L117)
- [token_configs.py:115-127](file://src/conf/tokenization/token_configs.py#L115-L127)
- [pretrain_mode.py:154-166](file://src/training/pretrain_mode.py#L154-L166)

## Core Components
- Structure vocabularies: tokens representing graph structure (node, edge, graph) and common special tokens
- Semantic vocabularies: tokens derived from node/edge/graph attributes (discrete and continuous)
- Reserved tokens: dataset/world-specific tokens used for semantics and structure
- Special tokens: mask, separator, and padding tokens
- Vocabulary persistence: saving/loading token-to-ID mappings to/from disk

Key responsibilities:
- Build unified vocabularies from configuration and dataset statistics
- Persist vocabularies to a file for reuse across workers and sessions
- Load vocabularies at runtime and map tokens to IDs during tokenization
- Support dynamic vocab updates via configuration and dataset augmentation

**Section sources**
- [vocab_builder.py:113-175](file://src/data/vocab_builder.py#L113-L175)
- [vocab_builder.py:178-218](file://src/data/vocab_builder.py#L178-L218)
- [tokenizer.py:70-82](file://src/data/tokenizer.py#L70-L82)
- [base.yaml:52-117](file://configs/tokenization/base.yaml#L52-L117)

## Architecture Overview
The vocabulary system integrates with the tokenization pipeline and training workflow:

```mermaid
sequenceDiagram
participant Trainer as "PretrainMode"
participant Builder as "VocabBuilder"
participant Tok as "GSTTokenizer"
participant Utils as "TokenizerUtils"
Trainer->>Builder : build_vocab(dataset, config, rank, use_cache)
Builder->>Builder : get_structure_vocab(config["structure"])
Builder->>Builder : get_semantics_vocab(dataset, config)
Builder->>Builder : save_vocab(vocab, fn)
Trainer->>Tok : instantiate with config and vocab_file
Tok->>Tok : load_vocab(fn) -> vocab_map
Trainer->>Tok : tokenize(graph)
Tok->>Tok : raw_tokenize -> sequence of tokens
Tok->>Tok : convert_tokens_to_ids -> input_dict
Tok->>Utils : prepare_inputs_for_task(...)
Utils-->>Tok : enhanced input_dict
Tok-->>Trainer : final batch-ready dict
```

**Diagram sources**
- [pretrain_mode.py:154-166](file://src/training/pretrain_mode.py#L154-L166)
- [vocab_builder.py:188-206](file://src/data/vocab_builder.py#L188-L206)
- [tokenizer.py:425-612](file://src/data/tokenizer.py#L425-L612)
- [tokenizer_utils.py:222-363](file://src/utils/tokenizer_utils.py#L222-L363)

## Detailed Component Analysis

### Vocabulary Construction and Persistence
- Structure vocabularies are generated from configuration-defined tokens for nodes, edges, graphs, and common special tokens.
- Semantic vocabularies are extracted from dataset attributes (discrete and continuous) and merged with reserved tokens and numbers.
- The builder supports caching and distributed workers by coordinating file existence and creation.

Concrete examples from the codebase:
- Building structure vocabularies: [get_structure_vocab:169-175](file://src/data/vocab_builder.py#L169-L175)
- Building semantic vocabularies: [get_semantics_vocab:85-110](file://src/data/vocab_builder.py#L85-L110)
- Saving vocabularies: [save_vocab:178-185](file://src/data/vocab_builder.py#L178-L185)
- Loading vocabularies: [load_vocab:209-218](file://src/data/vocab_builder.py#L209-L218)
- Distributed build coordination: [build_vocab:188-206](file://src/data/vocab_builder.py#L188-L206)

```mermaid
flowchart TD
Start(["Start build_vocab"]) --> CheckCache["Check cache and rank"]
CheckCache --> Exists{"Vocab exists?"}
Exists --> |Yes| Done["Skip build"]
Exists --> |No| MkDir["Ensure output directory"]
MkDir --> BuildStruct["Build structure vocab"]
BuildStruct --> BuildSemantics["Build semantics vocab"]
BuildSemantics --> Merge["Merge structure + semantics"]
Merge --> Save["Save to vocab file"]
Save --> Done
```

**Diagram sources**
- [vocab_builder.py:188-206](file://src/data/vocab_builder.py#L188-L206)

**Section sources**
- [vocab_builder.py:113-175](file://src/data/vocab_builder.py#L113-L175)
- [vocab_builder.py:178-218](file://src/data/vocab_builder.py#L178-L218)
- [vocab_builder.py:188-206](file://src/data/vocab_builder.py#L188-L206)

### Token Frequency Analysis and Vocabulary Expansion
- Discrete attributes are expanded into unique token identifiers per column and value, optionally sharing vocab across columns.
- Continuous attributes are decomposed into digit tokens (e.g., <5>, <0>, <3>) to form compact token sequences.
- The builder supports ignoring specific values and shuffling tokens for robustness.

Examples from the codebase:
- Discrete tokenization: [_tokenize_discrete_attr:688-717](file://src/data/tokenizer.py#L688-L717)
- Continuous tokenization: [_tokenize_continuous_attr:729-756](file://src/data/tokenizer.py#L729-L756)
- Ignored values and sharing vocab: [get_semantics_vocab:85-110](file://src/data/vocab_builder.py#L85-L110)

Practical guidance:
- Prefer shared vocab for discrete attributes when feasible to reduce vocabulary size.
- Use ignored values to filter rare or sentinel values.
- Shuffle tokens during tokenization to mitigate bias in early samples.

**Section sources**
- [vocab_builder.py:85-110](file://src/data/vocab_builder.py#L85-L110)
- [tokenizer.py:688-756](file://src/data/tokenizer.py#L688-L756)

### Reserved Token Categories
Reserved tokens are grouped into:
- Structure tokens: node, edge, graph, and common special tokens (mask, separator, reserved)
- Semantic tokens: dataset/world-specific reserved tokens and numeric tokens for continuous values
- Special tokens: padding, mask, separator, and optional instruction tokens

Configuration locations:
- Structure reserved tokens: [base.yaml:106-116](file://configs/tokenization/base.yaml#L106-L116)
- Semantic reserved tokens and numbers: [base.yaml:52-78](file://configs/tokenization/base.yaml#L52-L78)
- Tokenization config classes: [token_configs.py:106-127](file://src/conf/tokenization/token_configs.py#L106-L127)

Expansion techniques:
- Add new reserved tokens to configuration and rebuild vocabularies.
- Ensure reserved tokens are placed consistently across datasets to maintain alignment.

**Section sources**
- [base.yaml:52-117](file://configs/tokenization/base.yaml#L52-L117)
- [token_configs.py:106-127](file://src/conf/tokenization/token_configs.py#L106-L127)

### Token Mapping and Tokenization Pipeline
- Tokens are mapped to IDs using the loaded vocabulary map.
- The tokenizer prepares inputs for various tasks (pretraining, node, edge, graph) and supports packing multiple sequences.

Examples from the codebase:
- Token-to-ID mapping: [convert_tokens_to_ids:537-557](file://src/data/tokenizer.py#L537-L557)
- Task-specific input preparation: [prepare_inputs_for_task_*:222-363](file://src/utils/tokenizer_utils.py#L222-L363)
- EOS handling and padding: [pad/_pad_each_datapoint:227-357](file://src/data/tokenizer.py#L227-L357)

```mermaid
sequenceDiagram
participant Tok as "GSTTokenizer"
participant Map as "vocab_map"
participant Prep as "prepare_inputs_for_task_*"
Tok->>Map : vocab_map[token] for each token
Tok->>Tok : convert_tokens_to_ids -> input_ids, labels
Tok->>Prep : prepare_inputs_for_task(task_type, ...)
Prep-->>Tok : enhanced input_dict
```

**Diagram sources**
- [tokenizer.py:537-557](file://src/data/tokenizer.py#L537-L557)
- [tokenizer_utils.py:222-363](file://src/utils/tokenizer_utils.py#L222-L363)

**Section sources**
- [tokenizer.py:537-557](file://src/data/tokenizer.py#L537-L557)
- [tokenizer_utils.py:222-363](file://src/utils/tokenizer_utils.py#L222-L363)

### Configuration Options for Vocabulary File Formats and Pruning
- Vocabulary file format: a simple whitespace-separated token and ID per line
- Pruning strategies:
  - Ignore specific values for attributes
  - Share vocab across discrete attribute columns
  - Remove redundant edge-type tokens
- Dynamic updates:
  - Modify reserved tokens and rebuild vocab
  - Adjust scope_base and node_scope to control index token coverage

Examples from the codebase:
- File format: [save_vocab:178-185](file://src/data/vocab_builder.py#L178-L185)
- Ignoring values and sharing vocab: [get_semantics_vocab:85-110](file://src/data/vocab_builder.py#L85-L110)
- Removing edge-type tokens: [raw_tokenize:492-496](file://src/data/tokenizer.py#L492-L496)

**Section sources**
- [vocab_builder.py:178-185](file://src/data/vocab_builder.py#L178-L185)
- [vocab_builder.py:85-110](file://src/data/vocab_builder.py#L85-L110)
- [tokenizer.py:492-496](file://src/data/tokenizer.py#L492-L496)

### Relationships with Memory Optimization and Model Efficiency
- Vocabulary size directly affects embedding table size and memory footprint.
- Packing multiple sequences reduces overhead and improves throughput.
- Positional encoding and cyclic modes influence memory usage and training dynamics.

Evidence from the codebase:
- Packing sequences and positional IDs: [pack_token_seq:359-415](file://src/data/tokenizer.py#L359-L415)
- Positional ID computation and cyclic support: [get_input_dict_from_seq_tokens_id:639-685](file://src/data/tokenizer.py#L639-L685)
- Estimating tokens per sample for memory planning: [estimate_tokens_per_sample:181-196](file://src/training/pretrain_mode.py#L181-L196)

**Section sources**
- [tokenizer.py:359-415](file://src/data/tokenizer.py#L359-L415)
- [tokenizer.py:639-685](file://src/data/tokenizer.py#L639-L685)
- [pretrain_mode.py:181-196](file://src/training/pretrain_mode.py#L181-L196)

## Dependency Analysis
The vocabulary system interacts with tokenization and training modules:

```mermaid
graph LR
CFG["base.yaml"] --> TK["tokenizer.py"]
TCFG["token_configs.py"] --> TK
VB["vocab_builder.py"] --> TK
MU["mol_utils.py"] --> VB
NXU["nx_utils.py"] --> TK
PM["pretrain_mode.py"] --> VB
PM --> TK
```

**Diagram sources**
- [base.yaml:22-117](file://configs/tokenization/base.yaml#L22-L117)
- [token_configs.py:115-127](file://src/conf/tokenization/token_configs.py#L115-L127)
- [vocab_builder.py:188-218](file://src/data/vocab_builder.py#L188-L218)
- [tokenizer.py:30-120](file://src/data/tokenizer.py#L30-L120)
- [mol_utils.py:39-52](file://src/utils/mol_utils.py#L39-L52)
- [nx_utils.py:17-50](file://src/utils/nx_utils.py#L17-L50)
- [pretrain_mode.py:154-166](file://src/training/pretrain_mode.py#L154-L166)

**Section sources**
- [base.yaml:22-117](file://configs/tokenization/base.yaml#L22-L117)
- [token_configs.py:115-127](file://src/conf/tokenization/token_configs.py#L115-L127)
- [vocab_builder.py:188-218](file://src/data/vocab_builder.py#L188-L218)
- [tokenizer.py:30-120](file://src/data/tokenizer.py#L30-L120)
- [mol_utils.py:39-52](file://src/utils/mol_utils.py#L39-L52)
- [nx_utils.py:17-50](file://src/utils/nx_utils.py#L17-L50)
- [pretrain_mode.py:154-166](file://src/training/pretrain_mode.py#L154-L166)

## Performance Considerations
- Keep vocabulary size manageable to reduce embedding memory and improve training speed.
- Use packing to amortize fixed costs and increase effective batch utilization.
- Leverage ignored values and shared vocab to minimize redundancy.
- Monitor tokenization overhead and adjust scope_base/node_scope to balance coverage and size.

## Troubleshooting Guide
Common issues and resolutions:
- Out-of-vocabulary tokens: Ensure all tokens produced by tokenization exist in the vocabulary. Rebuild vocabularies after adding new reserved tokens or changing configurations.
- Vocabulary consistency across datasets: Align reserved tokens and numeric tokens across datasets to prevent mismatches.
- Version management: Store configuration files alongside vocabularies and version them to track changes in tokenization logic.

Evidence from the codebase:
- OOV handling and padding token: [load_vocab:209-218](file://src/data/vocab_builder.py#L209-L218)
- Reserved token alignment: [get_common_semantics/get_common_structure:119-123](file://src/data/tokenizer.py#L119-L123)

**Section sources**
- [vocab_builder.py:209-218](file://src/data/vocab_builder.py#L209-L218)
- [tokenizer.py:119-123](file://src/data/tokenizer.py#L119-L123)

## Conclusion
The vocabulary building system provides a structured, configurable approach to constructing graph token vocabularies. By combining structure, semantics, and special tokens, and by supporting caching, pruning, and dynamic updates, it enables efficient and scalable graph tokenization. Proper configuration and maintenance of vocabularies are essential for memory efficiency and model performance.

## Appendices

### Appendix A: Practical Examples from the Codebase
- Building and saving vocabularies: [build_vocab/save_vocab:188-206](file://src/data/vocab_builder.py#L188-L206)
- Loading vocabularies at runtime: [load_vocab:209-218](file://src/data/vocab_builder.py#L209-L218)
- Tokenization and input preparation: [GSTTokenizer methods:425-612](file://src/data/tokenizer.py#L425-L612), [prepare_inputs_for_task_*:222-363](file://src/utils/tokenizer_utils.py#L222-L363)
- Configuration examples: [base.yaml:22-117](file://configs/tokenization/base.yaml#L22-L117), [token_configs.py:115-127](file://src/conf/tokenization/token_configs.py#L115-L127)
- Dataset feature coverage for unified vocab: [read_complete_mol_features_ds/read_complete_onedevice_features_ds:39-52](file://src/utils/mol_utils.py#L39-L52)
