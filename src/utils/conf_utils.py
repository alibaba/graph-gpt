import os
import json
from typing import Dict
from pprint import pformat
from omegaconf import OmegaConf
from ..conf import TrainingConfig, Config


def parse_space_separated_args(args):
    """Convert space-separated key value pairs to dictionary"""
    config = {}
    i = 0
    while i < len(args):
        if " " in args[i] and args[i].count(" ") == 1:
            key, value = args[i].split(" ", 1)
            config[key] = value
            i += 1
        else:
            # Handle the case where key and value are separate
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                config[args[i]] = args[i + 1]
                i += 2
            else:
                config[args[i]] = True  # flag without value
                i += 1
    print(f"parsed conf:\n{config}")
    return config


def convert_to_legacy_tokenization_config(cfg: Config):
    token_cfg = cfg.tokenization
    train_cfg = cfg.training
    model_cfg = cfg.model

    tokenizer_config = OmegaConf.to_container(token_cfg, resolve=True)
    tokenizer_config.pop("data")
    tokenizer_config["name_or_path"] = os.path.join(
        token_cfg.data.data_dir, token_cfg.data.data_path
    )
    tokenizer_config["task_type"] = train_cfg.task_type
    tokenizer_config["pretrain_mlm"] = OmegaConf.to_container(
        train_cfg.pretrain_mlm, resolve=True
    )
    tokenizer_config["stacked_feat"] = model_cfg.graph_input.stacked_feat
    tokenizer_config["stack_method"] = model_cfg.graph_input.stack_method
    return tokenizer_config


def parse_deepspeed_config(training: TrainingConfig, loss_utils):
    batch_size = training.batch_size
    optim_cfg = training.optimizer
    lr = optim_cfg.lr
    min_lr = optim_cfg.lr
    warmup_num_steps = training.schedule.warmup_num_steps
    opt_config_dict = {}
    # parse deepspeed config
    with open(training.deepspeed_conf_file, "r") as f:
        ds_config = json.load(f)
    train_batch_size = (
        int(os.environ["WORLD_SIZE"])
        * batch_size
        * optim_cfg.gradient_accumulation_steps
    )
    ds_config["train_micro_batch_size_per_gpu"] = batch_size
    ds_config["gradient_accumulation_steps"] = optim_cfg.gradient_accumulation_steps
    ds_config["train_batch_size"] = train_batch_size
    if optim_cfg.max_grad_norm > 0:
        ds_config["gradient_clipping"] = optim_cfg.max_grad_norm
    if "optimizer" in ds_config.keys():
        ds_config["optimizer"]["params"]["lr"] = lr
        ds_config["optimizer"]["params"]["betas"] = list(optim_cfg.betas)
        ds_config["optimizer"]["params"]["eps"] = optim_cfg.eps
        ds_config["optimizer"]["params"]["weight_decay"] = optim_cfg.weight_decay
        ds_config = update_deepspeed_config(opt_config_dict, ds_config)
        if ("scheduler" in ds_config.keys()) and (
            ds_config["scheduler"]["type"] in loss_utils.DS_SCHEDULER_LS
        ):
            ds_config["scheduler"]["params"] = loss_utils.set_ds_scheduler(
                ds_config["scheduler"]["type"],
                ds_config["scheduler"]["params"],
                # for WarmupLR & WarmupDecayLR
                warmup_max_lr=lr,
                warmup_min_lr=min_lr,
                warmup_num_steps=warmup_num_steps,
                total_num_steps=training.schedule.total_num_steps,
                # for LRRangeTest
                lr_range_test_min_lr=min_lr,
                lr_range_test_step_size=warmup_num_steps,
                # for OneCycle
                cycle_min_lr=min_lr,
                cycle_max_lr=lr,
                cycle_first_step_size=warmup_num_steps,
            )
            ds_config["scheduler"]["params"]["last_batch_iteration"] = -1
    if "zero_optimization" in ds_config.keys():
        if ds_config["zero_optimization"]["stage"] == 0:
            ds_config["zero_optimization"].pop("offload_optimizer")
    if "tensorboard" in ds_config.keys():
        ds_config["tensorboard"]["output_path"] = training.output_dir
    ds_config["flops_profiler"]["output_file"] = os.path.join(
        training.output_dir, "flops_profile.txt"
    )
    return ds_config


def parse_deepspeed_config_for_ft(train_cfg: TrainingConfig, loss_utils):
    optim_cfg = train_cfg.optimizer
    ds_config = parse_deepspeed_config(train_cfg, loss_utils)
    print(f"\nds_config:\n{pformat(ds_config)}")
    if ("scheduler" in ds_config.keys()) and (
        ds_config["scheduler"]["type"] not in loss_utils.DS_SCHEDULER_LS
    ):
        non_ds_scheduler, scheduler_conf = loss_utils.set_py_scheduler(
            ds_config["scheduler"]["type"],
            ds_config,
            # for CyclicLR
            base_lr=optim_cfg.min_lr,
            max_lr=optim_cfg.lr,
            step_size_up=train_cfg.schedule.warmup_num_steps,
            # for CosineAnnealingLR
            T_max=train_cfg.schedule.total_num_steps,
            eta_min=optim_cfg.min_lr,
            # for CosineAnnealingWarmRestarts
            T_0=train_cfg.schedule.warmup_num_steps,
            # for OneCycleLR
            total_steps=train_cfg.schedule.total_num_steps,
            pct_start=train_cfg.schedule.warmup_num_steps
            / train_cfg.schedule.total_num_steps,
            min_lr=optim_cfg.min_lr,
            # for general
            last_step_index=-1,
        )
    else:
        non_ds_scheduler, scheduler_conf = None, {}
    return ds_config, non_ds_scheduler, scheduler_conf


def update_deepspeed_config(opt_config: Dict, ds_config: Dict):
    # update ds_config with opt_config parameters
    if opt_config:
        ds_config["optimizer"]["params"].update(opt_config["optimizer"]["params"])
        ds_config["gradient_clipping"] = opt_config["gradient_clipping"]
        ds_config["scheduler"]["params"]["warmup_max_lr"] = opt_config["scheduler"][
            "params"
        ]["warmup_max_lr"]
        ds_config["scheduler"]["type"] = opt_config["scheduler"]["type"]
    return ds_config


def init_log_conf(
    misc_utils,
    pretrain_cpt,
    output_dir,
    steps_per_saving,
):
    if pretrain_cpt == output_dir:
        _, prev_epoch = misc_utils.get_latest_ckp(pretrain_cpt)
        last_step_index = prev_epoch * steps_per_saving
        ls_log, ls_result, _ = misc_utils.load_all(output_dir, load_result=True)
        last_step_index_from_log = int(ls_log[-1].strip().split(",")[2])
        assert (
            last_step_index >= last_step_index_from_log
        ), f"last_step_index: {last_step_index}, last_step_index_from_log: {last_step_index_from_log}"
        print(
            f"Resume training from {pretrain_cpt} with last_step_index {last_step_index}!"
        )
        ep_init = prev_epoch
        j_init = last_step_index
    else:
        last_step_index = -1
        ep_init = 0
        j_init = 0
        ls_log = ["epoch,lr,local_step,global_step,train_loss\n"]
        ls_result = ["epoch,global_step,acc\n"]
    return last_step_index, ep_init, j_init, ls_log, ls_result


def init_log_conf_for_ft(
    misc_utils,
    pretrain_cpt,
    output_dir,
    steps_per_epoch,
    eval_only,
):
    if pretrain_cpt == output_dir:
        _, prev_epoch = misc_utils.get_latest_ckp(pretrain_cpt, eval_only)
        last_step_index = (prev_epoch + 1) * steps_per_epoch
        ls_log, ls_result, ls_loss = misc_utils.load_all(
            output_dir, load_log=True, load_result=True, load_loss=True
        )

        last_step_index_from_log = int(ls_log[-1].strip().split(",")[2])
        if not eval_only:
            assert (
                last_step_index >= last_step_index_from_log
            ), f"last_step_index: {last_step_index}, last_step_index_from_log: {last_step_index_from_log}"

        last_step_index_from_result = int(ls_result[-1].strip().split(",")[1])
        if not eval_only:
            assert (
                last_step_index == last_step_index_from_result
            ), f"last_step_index: {last_step_index}, last_step_index_from_result: {last_step_index_from_result}"

        last_step_index_from_loss = int(ls_loss[-1].strip().split(",")[2])
        if not eval_only:
            assert (
                last_step_index >= last_step_index_from_loss
            ), f"last_step_index: {last_step_index}, last_step_index_from_loss: {last_step_index_from_loss}"

        print(
            f"Resume training from {pretrain_cpt} with last_step_index {last_step_index}!"
        )
        ep_init = prev_epoch + 1
        j_init = last_step_index
    else:
        last_step_index = -1
        ep_init = 0
        j_init = 0
        ls_log = ["epoch,local_step,global_step,train_loss,test_loss,metrics\n"]
        ls_result = [
            "epoch,global_step,all_lrs,lr,train_metric,valid_metric,test_metrics\n"
        ]
        ls_loss = ["epoch,local_step,global_step,ntp_loss,task_loss,train_loss\n"]
    # if last_step_index == total_num_steps:
    #     print(
    #         f"[WARNING] last_step_index == total_num_steps == {last_step_index}! SET to -1! lr RESTARTED!"
    #     )
    #     last_step_index = -1
    # if got `KeyError: "param 'initial_lr' is not specified`
    # refer: https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822/5
    return last_step_index, ep_init, j_init, ls_log, ls_result, ls_loss
