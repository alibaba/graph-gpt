# Changelog

All notable changes to this project will be documented in this file.

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
