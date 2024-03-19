# GraphGPT: Graph Learning with Generative Pre-trained Transformers

This repository is the official implementation of “[GraphGPT: Graph Learning with Generative Pre-trained Transformers](https://arxiv.org/abs/2401.00529)” in [PyTorch](https://github.com/pytorch/pytorch).

> GraphGPT: Graph Learning with Generative Pre-trained Transformers
>
> Qifang Zhao, Weidong Ren, Tianyu Li, Xiaoxiao Xu, Hong Liu

## Update:

***03/19/2024***
  1. v0.2.0 released.
  2. Implement `permute_nodes` for graph-level map-style dataset, in order to increase variations of Eulerian paths,
     and result in better and robust results.
  3. Add `StackedGSTTokenizer` so that semantics (i.e., node/edge attrs) tokens can be stacked together with structural 
     tokens, and the length of sequence would be reduced a lot.
  4. refactor codes.

***01/23/2024***
  1. v0.1.1, fix bugs of common-io package.

***01/03/2024***
  1. Initial release of codes.

## Overview:
![Alt text](pic/architect.png?raw=true "Model Overview")

GraphGPT is a novel model for Graph learning by self-supervised Generative Pre-training Transformers.
Our model transforms each graph or sampled subgraph into a sequence of tokens representing the node,
edge and attributes reversibly using the Eulerian path first.
Then we feed the tokens into a standard transformer decoder and pre-train it with the next-token-prediction (NTP) task.
Lastly, we fine-tune the GraphGPT model with the supervised tasks.
This intuitive, yet effective model achieves superior or close results to the state-of-the-art methods
for the graph-, edge- and node-level tasks on the large scale molecular dataset PCQM4Mv2,
the protein-protein association dataset ogbl-ppa and the ogbn-proteins dataset from the Open Graph Benchmark (OGB).
Furthermore, the generative pre-training enables us to train GraphGPT up to 400M+ parameters
with consistently increasing performance, which is beyond the capability of GNNs and previous graph transformers.
The source code and pre-trained checkpoints will be released in this repository to pave the way for the
graph foundation model research, and also to assist the scientific discovery in pharmaceutical,
chemistry, material and bio-informatics domains, etc.


## Results

### Graph-level-task: PCQM4M-v2 dataset

![](pic/graph-lvl-exp.png)

### Edge-level-task: ogbl-ppa dataset

![](pic/edge-lvl-exp.png)

### Node-level-task: ogbn-proteins dataset

![](pic/node-lvl-exp.png)

## Installation

- Clone this repository

```shell
git clone https://github.com/alibaba/graph-gpt.git
```

- Install the dependencies in requirements.txt (Using [Anaconda](https://www.anaconda.com/), tested with py38, pytorch-1131 and CUDA-11.7, 11.8 and 12.1 on GPU V100 and A100)

```shell
conda create -n graph_gpt python=3.8 pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda activate graph_gpt
cd graph-gpt
pip install -r ./requirements.txt
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
sudo apt-get install bc
```


## Datasets

The datasets are downloaded using python package [ogb](https://pypi.org/project/ogb/).

When you run scripts in `./examples`, the dataset will be automatically downloaded.

However, the dataset PCQM4M-v2 is huge, and downloading and
preprocessing might be problematic. We suggest `cd ./src/utils/` and `python dataset_utils.py`
to download and preprocess dataset separately.


## Run

1. Modify parameters in `./examples/ggpt_pretrain.sh`, e.g., `dataset_name`, `model_name`,
  `batch_size`, `workerCount` and etc, and then run `./examples/ggpt_pretrain.sh` to pretrain
  the model with the dataset.
2. Modify parameters in `./examples/ggpt_supervised.sh`, e.g., `dataset_name`, `model_name`,
  `batch_size`, `workerCount`, `pretrain_cpt` and etc, and then run `./examples/ggpt_supervised.sh`
  to fine-tune with downstream tasks.


## Code Norm
### Pre-commit
- Check the [official website](https://pre-commit.com/) for details
- `.pre-commit-config.yaml`: create the file with following content for python
  - ```yaml
    repos:
    -   repo: https://github.com/pre-commit/pre-commit-hooks
        rev: v4.4.0
        hooks:
        -   id: check-yaml
        -   id: end-of-file-fixer
        -   id: trailing-whitespace
    -   repo: https://github.com/psf/black
        rev: 23.7.0
        hooks:
        -   id: black
    ```
- `pre-commit install`: install pre-commit into your git hooks.
  - pre-commit will now run on every commit.
  - Every time you clone a project using pre-commit running `pre-commit install` should always be the first thing you do.
- `pre-commit run --all-files`: run all pre-commit hooks on a repository
- `pre-commit autoupdate`: update your hooks to the latest version automatically
- `git commit -n`: pre-commit checks can be disabled for a particular commit with the command


## Citation

If you find this work useful, please kindly cite following papers:

```latex
@article{zhao2024graphgpt,
  title={GraphGPT: Graph Learning with Generative Pre-trained Transformers},
  author={Zhao, Qifang and Ren, Weidong and Li, Tianyu and Xu, Xiaoxiao and Liu, Hong},
  journal={arXiv preprint arXiv:2401.00529},
  year={2024}
}
```

## Contact

Qifang Zhao (james.zqf@alibaba-inc.com)

Sincerely appreciate your suggestions on our work!

## License

Released under the MIT license (see `LICENSE`):

```text
Ali-GraphGPT-project is an AI project on training large scale transformer decoder with graph datasets,
developed by Alibaba and licensed under the MIT License.
```
