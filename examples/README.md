# How to Run

This folder contains examples of running GraphGPT in local machine with multiple GPUs.
We adopt `deepspeed` engine to run the script. In Alibaba Cloud PAI platform,
`python -m torch.distributed.launch` is used to run the parallel training.

# Test Procedure when new version is released

When new version of code is developed, it has to pass tests to be released.
The current procedure is listed below.

1. Delete the existing vocab files of the datasets
2. Run `ggpt_pretrain.sh` and `ggpt_supervised.sh` for the datasets `ogbn-proteins`
    `ogbl-ppa`, `ogbg-molpcba` and `PCQM4M-v2` with `mini` architecture quickly, make sure no
    bugs occur. This step does not need to produce the best results.
3. Repeat step 2, but this time run long enough for each dataset, so that best results
    can be re-produced.
