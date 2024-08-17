import os
import json
from typing import Dict
from datetime import datetime
from pprint import pprint, pformat


def parse_tokenization_config(
    tokenization_config,
    pretrain_cpt,
    data_dir,
    dataset_name,
    task_type,
    tokenizer_class,
    attr_assignment,
    **kwargs,
):
    with open(tokenization_config, "r") as fp:
        tokenizer_config = json.load(fp)
        nx_config = tokenizer_config["structure"].get("nx", {})
    if len(pretrain_cpt) > 0:
        print(f"Ignoring input tokenization config file\n{tokenization_config}")
        tokenization_config = os.path.join(pretrain_cpt, "tokenization_config.json")
        print(f"Use saved pretrain tokenization config file\n{tokenization_config}\n")
    with open(tokenization_config, "r") as fp:
        tokenizer_config = json.load(fp)
        tokenizer_config["data_dir"] = data_dir
        tokenizer_config["name_or_path"] = (
            os.path.join(data_dir, tokenizer_config["name_or_path"])
            if len(pretrain_cpt) == 0
            else tokenizer_config["name_or_path"]
        )
        tokenizer_config["dataset"] = dataset_name
        tokenizer_config["task_type"] = task_type
        if tokenizer_class is not None:
            tokenizer_config["tokenizer_class"] = tokenizer_class
        tokenizer_config["structure"]["nx"] = nx_config
    if len(attr_assignment) > 0:
        tokenizer_config["semantics"]["attr_assignment"] = attr_assignment
    if tokenizer_config["tokenizer_class"] == "StackedGSTTokenizer":
        tokenizer_config["semantics"]["node"]["ignored_val"] = None
        tokenizer_config["semantics"]["edge"]["ignored_val"] = None
        tokenizer_config["semantics"]["attr_shuffle"] = False
    return tokenizer_config


def parse_tokenization_config_for_ft(
    tokenization_config,
    pretrain_cpt,
    data_dir,
    dataset_name,
    task_type,
    tokenizer_class,
    attr_assignment,
    attr_shuffle,
):
    tokenizer_config = parse_tokenization_config(
        tokenization_config=tokenization_config,
        pretrain_cpt=pretrain_cpt,
        data_dir=data_dir,
        dataset_name=dataset_name,
        task_type=task_type,
        tokenizer_class=tokenizer_class,
        attr_assignment=attr_assignment,
    )
    with open(tokenization_config, "r") as fp:
        tmp_config = json.load(fp)
        sampling_config = tmp_config["sampling"]
    tokenizer_config["sampling"] = sampling_config
    tokenizer_config["pretrain_cpt"] = pretrain_cpt
    tokenizer_config["semantics"]["graph"] = {
        "discrete": None,
        "continuous": None,
        "ignored_val": None,
    }
    tokenizer_config["semantics"]["add_3d"] = False
    tokenizer_config["semantics"]["attr_shuffle"] = bool(attr_shuffle)
    tokenizer_config["semantics"]["attr_mask_ratio"] = 0
    tokenizer_config["semantics"]["instructions"] = {}
    tokenizer_config["structure"]["nx"] = {}
    return tokenizer_config


def parse_model_config(
    model_config,
    pretrain_cpt,
    GraphModelConfig,
    gtokenizer,
    hidden_size,
    intermediate_size,
    num_hidden_layers,
    num_attention_heads,
    hidden_act,
    max_position_embeddings,
    initializer_range,
    rope_theta,
    tie_word_embeddings,
    causal_attention,
    attention_dropout,
    embed_dropout,
    path_dropout,
    mlp_dropout,
    layer_scale_init_value,
    next_n_token,
    stacked_feat,
    stacked_feat_agg_method,
    **kwargs,
):
    if len(model_config) > 0:
        with open(model_config, "r") as fp:
            model_config_dict = json.load(fp)
            print(
                f"[{datetime.now()}] Load model config\n{pformat(model_config_dict)}\nfrom\n{model_config}\n"
            )
    else:
        model_config_dict = {}
    if len(pretrain_cpt) > 0:
        model_config = os.path.join(pretrain_cpt, "config.json")
        print(
            f"[{datetime.now()}] Use saved pretrain model config file\n{model_config}\n"
        )
        config = GraphModelConfig().from_pretrained(model_config)
    else:
        config = GraphModelConfig()
    config.update(model_config_dict)
    config.update(
        {
            "vocab_size": gtokenizer.vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_attention_heads,
            "hidden_act": hidden_act,
            "max_position_embeddings": max_position_embeddings,
            "initializer_range": initializer_range,
            "rope_theta": rope_theta,
            "tie_word_embeddings": bool(tie_word_embeddings),
            "bos_token_id": gtokenizer.get_bos_token_id(),
            "eos_token_id": gtokenizer.get_eos_token_id(),
            "causal_attention": bool(causal_attention),
            "attention_dropout": attention_dropout,
            "embed_pdrop": embed_dropout,
            "path_pdrop": path_dropout,
            "mlp_pdrop": mlp_dropout,
            "layer_scale_init_value": layer_scale_init_value,
            "next_n_token": next_n_token,
            "stacked_feat": stacked_feat,
            "stacked_feat_agg_method": stacked_feat_agg_method,
        }
    )
    return config


def parse_model_config_for_ft(
    model_config,
    pretrain_cpt,
    GraphModelConfig,
    gtokenizer,
    hidden_size,
    intermediate_size,
    num_hidden_layers,
    num_attention_heads,
    hidden_act,
    max_position_embeddings,
    initializer_range,
    # rope_theta,
    # tie_word_embeddings,
    causal_attention,
    attention_dropout,
    embed_dropout,
    path_dropout,
    mlp_dropout,
    layer_scale_init_value,
    # next_n_token,
    stacked_feat,
    stacked_feat_agg_method,
    num_labels,
    mlp,
    pooling_method,
    dropout,
    problem_type,
    loss_type,
    loss_utils,
    tokenizer_config,
    ntp_ratio,
    **kwargs,
):
    if len(pretrain_cpt) > 0:
        tmp_model_config = os.path.join(pretrain_cpt, "config.json")
        tmp_config = GraphModelConfig().from_pretrained(tmp_model_config)
        num_attention_heads = tmp_config.num_attention_heads
        num_key_value_heads = tmp_config.num_key_value_heads
        stacked_feat = tmp_config.stacked_feat
        stacked_feat_agg_method = tmp_config.stacked_feat_agg_method
        rope_theta = tmp_config.rope_theta
    else:
        num_key_value_heads = num_attention_heads
        rope_theta = 10000
    tie_word_embeddings = False
    next_n_token = 1

    config = parse_model_config(
        model_config=model_config,
        pretrain_cpt=pretrain_cpt,
        GraphModelConfig=GraphModelConfig,
        gtokenizer=gtokenizer,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        hidden_act=hidden_act,
        max_position_embeddings=max_position_embeddings,
        initializer_range=initializer_range,
        rope_theta=rope_theta,
        tie_word_embeddings=tie_word_embeddings,
        causal_attention=causal_attention,
        attention_dropout=attention_dropout,
        embed_dropout=embed_dropout,
        path_dropout=path_dropout,
        mlp_dropout=mlp_dropout,
        layer_scale_init_value=layer_scale_init_value,
        next_n_token=next_n_token,
        stacked_feat=stacked_feat,
        stacked_feat_agg_method=stacked_feat_agg_method,
    )

    config.update(
        {
            "num_key_value_heads": num_key_value_heads,
            # below for downstream fine-tune task
            "num_labels": num_labels,
            "mlp": mlp,
            "pooling_method": pooling_method,
            "dropout": dropout,
            "problem_type": problem_type,
            "loss_type": loss_type if len(loss_type) > 0 else None,
            "num_neg": loss_utils.get_neg_ratio(tokenizer_config["sampling"]),
            "use_ntp": ntp_ratio > 0,
        }
    )
    return config


def parse_deepspeed_config(
    optimization_config,
    deepspeed_config,
    batch_size,
    gradient_accumulation_steps,
    max_grad_norm,
    lr,
    betas,
    eps,
    weight_decay,
    min_lr,
    total_num_steps,
    warmup_num_steps,
    output_dir,
    loss_utils,
    **kwargs,
):
    # parse optimization_config file
    if len(optimization_config) > 0:
        with open(optimization_config, "r") as fp:
            opt_config_dict = json.load(fp)
            print(
                f"[{datetime.now()}] Load optimization config {pformat(opt_config_dict)} from {optimization_config}"
            )
    else:
        opt_config_dict = {}
    # parse deepspeed config
    with open(deepspeed_config, "r") as f:
        ds_config = json.load(f)
    train_batch_size = (
        int(os.environ["WORLD_SIZE"]) * batch_size * gradient_accumulation_steps
    )
    ds_config["train_micro_batch_size_per_gpu"] = batch_size
    ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps
    ds_config["train_batch_size"] = train_batch_size
    if max_grad_norm > 0:
        ds_config["gradient_clipping"] = max_grad_norm
    if "optimizer" in ds_config.keys():
        ds_config["optimizer"]["params"]["lr"] = lr
        ds_config["optimizer"]["params"]["betas"] = betas
        ds_config["optimizer"]["params"]["eps"] = eps
        ds_config["optimizer"]["params"]["weight_decay"] = weight_decay
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
                total_num_steps=total_num_steps,
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
        ds_config["tensorboard"]["output_path"] = output_dir
    return ds_config


def parse_deepspeed_config_for_ft(
    optimization_config,
    deepspeed_config,
    batch_size,
    gradient_accumulation_steps,
    max_grad_norm,
    lr,
    betas,
    eps,
    weight_decay,
    min_lr,
    total_num_steps,
    warmup_num_steps,
    output_dir,
    loss_utils,
    **kwargs,
):
    ds_config = parse_deepspeed_config(
        optimization_config=optimization_config,
        deepspeed_config=deepspeed_config,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        min_lr=min_lr,
        total_num_steps=total_num_steps,
        warmup_num_steps=warmup_num_steps,
        output_dir=output_dir,
        loss_utils=loss_utils,
    )
    print(f"\nds_config:\n{pformat(ds_config)}")
    if ("scheduler" in ds_config.keys()) and (
        ds_config["scheduler"]["type"] not in loss_utils.DS_SCHEDULER_LS
    ):
        non_ds_scheduler, scheduler_conf = loss_utils.set_py_scheduler(
            ds_config["scheduler"]["type"],
            ds_config,
            # for CyclicLR
            base_lr=min_lr,
            max_lr=lr,
            step_size_up=warmup_num_steps,
            # for CosineAnnealingLR
            T_max=total_num_steps,
            eta_min=min_lr,
            # for CosineAnnealingWarmRestarts
            T_0=warmup_num_steps,
            # for OneCycleLR
            total_steps=total_num_steps,
            pct_start=warmup_num_steps / total_num_steps,
            min_lr=min_lr,
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


def dump_all_conf(
    output_dir,
    tokenizer_config,
    params,
    tmp_ds_config,
    use_deepspeed=True,
    **kwargs,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "tokenization_config.json"), "w+") as fp:
        json.dump(tokenizer_config, fp, indent=4)
    print(f"[{datetime.now()}] Finish -> Dump to `tokenization_config.json`")
    with open(os.path.join(output_dir, "params.txt"), "w+") as fp:
        fp.write(params)
    print(f"[{datetime.now()}] Finish -> Dump to `params.txt`")
    if use_deepspeed:
        with open(os.path.join(output_dir, "ds_config.json"), "w+") as fp:
            json.dump(tmp_ds_config, fp, indent=4)
        print(f"[{datetime.now()}] Finish -> Dump to `ds_config.json`")


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
):
    if pretrain_cpt == output_dir:
        _, prev_epoch = misc_utils.get_latest_ckp(pretrain_cpt)
        last_step_index = (prev_epoch + 1) * steps_per_epoch
        ls_log, ls_result, ls_loss = misc_utils.load_all(
            output_dir, load_log=True, load_result=True, load_loss=True
        )

        last_step_index_from_log = int(ls_log[-1].strip().split(",")[2])
        assert (
            last_step_index >= last_step_index_from_log
        ), f"last_step_index: {last_step_index}, last_step_index_from_log: {last_step_index_from_log}"

        last_step_index_from_result = int(ls_result[-1].strip().split(",")[1])
        assert (
            last_step_index == last_step_index_from_result
        ), f"last_step_index: {last_step_index}, last_step_index_from_result: {last_step_index_from_result}"

        last_step_index_from_loss = int(ls_loss[-1].strip().split(",")[2])
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
