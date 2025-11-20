import os
import json
import copy
from datetime import datetime
from pprint import pprint
from typing import Dict
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_utils import PreTrainedModel
from torchmetrics.classification import Accuracy
from omegaconf import OmegaConf
from ..conf import (
    Config,
    TrainingConfig,
    GenerationConfig,
    TrainingStats,
    OptimizingStats,
    EMAStats,
    LoaderStats,
)
from . import (
    misc_utils,
    metrics_utils,
    get_metrics,
    ogb_utils,
    evaluate_ogb,
    loader_utils,
    format_ogb_output_for_csv,
    generation_utils,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter


@torch.no_grad()
def ft_infer(model, loader, cfg: Config, eval_name: str):
    # Infer hidden states or predicted results for a SFT model
    # eval_name -> train, valid, test
    model.eval()
    device = model.device
    ls_idx = []
    ls_logits = []
    ls_states = []
    for j, test_data in enumerate(loader, 1):
        input_ids = test_data["input_ids"].to(device)
        attention_mask = test_data["attention_mask"].to(device)
        position_ids = test_data["position_ids"].to(device)
        inputs_raw_embeds = None
        if "embed" in test_data:
            inputs_raw_embeds = test_data["embed"].to(device)
        res = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_raw_embeds=inputs_raw_embeds,
            position_ids=position_ids,
        )  # Perform a single forward pass.
        # `idx` is finalized inside `collator`
        idx = test_data["idx"].to(device)
        ls_idx.append(idx.view((-1, 1)))
        ls_logits.append(res.task_logits.half())
        if cfg.training.ft_eval.save_hidden_states:
            ls_states.append(res.task_hidden_states.half())
    idx = torch.vstack(ls_idx)
    logits = torch.vstack(ls_logits)
    if cfg.training.ft_eval.save_hidden_states:
        hidden_states = torch.vstack(ls_states)
    else:
        hidden_states = None
    return idx, logits, hidden_states


@torch.no_grad()
def ft_evaluate(model, loader, cfg: Config, eval_name: str):
    # Evaluate sft performance
    # eval_name -> train, valid, test
    problem_type = cfg.model.ft_head.problem_type
    num_labels = cfg.model.ft_head.num_labels
    task_level = cfg.model.ft_head.task_type
    metric_type = cfg.model.ft_head.metric_type
    dataset_name = cfg.tokenization.data.dataset
    model.eval()
    device = model.device
    # model.train()
    # print(f"Enabling dropout during inference ... ")
    # if model.embed_dropout is not None:
    #     model.embed_dropout.train()  # no use for pcqm4m-v2
    #     print(f"Enabling embed dropout: {model.embed_dropout} during inference ... ")
    if not metric_type:
        metric_type = problem_type
    cls_metrics = get_metrics(metric_type, device, num_labels=num_labels)
    test_loss = 0
    for j, test_data in enumerate(loader, 1):
        input_ids = test_data["input_ids"].to(device)
        attention_mask = test_data["attention_mask"].to(device)
        position_ids = test_data["position_ids"].to(device)
        task_labels = test_data[f"{task_level}_labels"].to(device)
        task_labels = (
            task_labels.float()
            if problem_type == "multi_label_classification"
            else task_labels
        )
        pretrain_labels = test_data["labels"].to(device)
        cls_idx = test_data["cls_idx"].to(device) if "cls_idx" in test_data else None
        inputs_raw_embeds = None
        if "embed" in test_data:
            inputs_raw_embeds = test_data["embed"].to(device)
        sample_wgt = None
        if "wgt" in test_data:
            sample_wgt = test_data["wgt"].to(device)
        res = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_labels=task_labels,
            pretrain_labels=pretrain_labels,
            cls_idx=cls_idx,
            inputs_raw_embeds=inputs_raw_embeds,
            sample_wgt=sample_wgt,
            position_ids=position_ids,
        )  # Perform a single forward pass.
        # record loss
        test_loss += res.task_loss
        # record metrics
        # `idx` is finalized inside `collator`
        idx = test_data["idx"].to(device)
        if "raw_node_idx" in test_data:
            idx = (idx, test_data["raw_node_idx"].to(device))
        cls_metrics.update(res.task_logits, task_labels, idx)
    print(f"[{datetime.now()}][FINISHED][EVAL] input_ids shape: {input_ids.shape}")
    test_loss = test_loss / j
    cls_metrics.compute()
    ogb_input_dict = cls_metrics.to_dict()
    try:
        world_size = dist.get_world_size()
    except ValueError:
        print("In local test setting!!!\n" * 5)
        world_size = 1
    if world_size > 1:
        print(f"[{datetime.now()}][GATHERING] start gathering result tensors ...")
        # gather results from different gpus if there are more than 1 gpu
        # refer to: https://stackoverflow.com/a/71433508/4437068
        for key, val in ogb_input_dict.items():
            ogb_input_dict[key] = tensor_out = misc_utils.all_gather(val.to(device))
            cnt_nans = torch.isnan(tensor_out).sum().item()
            print(
                f"[{datetime.now()}] Update ogb_input_dict element {key} with val of shape {tensor_out.shape} of type {tensor_out.dtype} of NANs {cnt_nans}"
            )

    if dataset_name in ogb_utils._eval._register_map:
        if (eval_name == "train") and (
            dataset_name in {"ogbl-citation2", "ogbl-wikikg2"}
        ):
            # ONLY these ds' valid/test shall be evaluated
            ogb_eval_res = None
        else:
            ogb_eval_res = evaluate_ogb(dataset_name, ogb_input_dict)
    else:
        ogb_eval_res = cls_metrics.results_in_dict()
    return test_loss, cls_metrics, ogb_eval_res, ogb_input_dict


@torch.no_grad()
def evaluate(
    model: PreTrainedModel,
    loader: DataLoader,
    eval_name: str = "valid",  # valid or test
    do_eval: bool = True,
):
    if not do_eval:
        return None, None
    try:
        world_size = dist.get_world_size()
        rank = int(dist.get_rank())
    except ValueError:
        print("In local machine test setting!!!\n" * 5)
        world_size = 1
        rank = 0
    model.eval()
    device = model.device
    print(f"[{datetime.now()}] Evaluating Model on {eval_name} data ...")
    ls_loss = []
    ls_aux_loss = []
    for test_data in loader:
        input_ids = test_data["input_ids"].to(device)
        attention_mask = test_data["attention_mask"].to(device)
        position_ids = test_data["position_ids"].to(device)
        labels = test_data["labels"].to(device)
        inputs_raw_embeds = None
        if "embed" in test_data:
            inputs_raw_embeds = test_data["embed"].to(device)
        sample_wgt = None
        if "wgt" in test_data:
            sample_wgt = test_data["wgt"].to(device)
        res = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            inputs_raw_embeds=inputs_raw_embeds,
            # position_ids=position_ids,
            sample_wgt=sample_wgt,
        )  # Perform a single forward pass.
        loss = res.head1_loss
        aux_loss = res.head2_loss
        if world_size > 1:
            dist.reduce(loss, 0)
            if aux_loss is not None:
                dist.reduce(aux_loss, 0)
            if rank == 0:
                loss = loss / world_size
                if aux_loss is not None:
                    aux_loss = aux_loss / world_size
        ls_loss.append(loss)
        if aux_loss is not None:
            if not torch.isnan(aux_loss).item():
                ls_aux_loss.append(aux_loss)

    loss = sum(ls_loss) / len(ls_loss)
    if len(ls_aux_loss) > 0:
        aux_loss = sum(ls_aux_loss) / len(ls_aux_loss)
    print(
        f"[eval mode] input_ids: {input_ids.shape}, loss: {loss}, aux_loss: {aux_loss}"
    )
    model.train()
    return loss, None


@torch.no_grad()
def evaluate_generation(
    model: PreTrainedModel,
    loader: DataLoader,
    eval_name: str = "valid",  # valid or test
    do_eval: bool = True,
    cfg: Config = None,
):
    if not do_eval:
        return ""
    try:
        world_size = dist.get_world_size()
    except ValueError:
        print("In local machine test setting!!!\n" * 5)
        world_size = 1
    model.eval()
    device = model.device
    gen_cfg = cfg.generation
    train_cfg = cfg.training
    gen_cfg.output_history = False
    assert gen_cfg.alg in ["origin", "maskgit_plus", "topk_margin", "entropy"]
    default_umr_clip = list(train_cfg.pretrain_mlm.params.umr_clip)
    print(f"[{datetime.now()}] Evaluating Model Generation on {eval_name} data ...")
    print(OmegaConf.to_yaml(gen_cfg))
    intervals = 5
    sp = np.linspace(0.49, 0.99, num=intervals + 1)
    acc_mat = []
    for j in range(intervals):
        # `j in range(intervals)` must be in the outer loop to affect loader's processing of data mask
        train_cfg.pretrain_mlm.params.umr_clip = [float(sp[j]), float(sp[j + 1])]
        print(
            f"[{datetime.now()}] j: {j}, umr_clip: {train_cfg.pretrain_mlm.params.umr_clip}"
        )
        for test_data in loader:
            input_ids = test_data["input_ids"].to(device)
            attention_mask = test_data["attention_mask"].to(device)
            labels = test_data["labels"].to(device)
            inputs_raw_embeds = None
            if "embed" in test_data:
                inputs_raw_embeds = test_data["embed"].to(device)

            if gen_cfg.parallel_gen:
                eval_func = eval_gen_per_batch
            else:
                eval_func = eval_gen_per_sample
            acc = eval_func(
                model,
                gen_cfg,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                inputs_raw_embeds=inputs_raw_embeds,
            )
            acc_mat.append(acc)
    acc_mat = torch.hstack(acc_mat).reshape((intervals, -1))
    acc_mat = acc_mat.T  # [seq, intervals]
    acc_sum = acc_mat.float().sum(dim=0)
    num_samp = torch.tensor([acc_mat.shape[0]], dtype=acc_sum.dtype, device=device)
    # TODO: take average before all_gather to avoid possible overflow

    if world_size > 1:
        acc_sum = misc_utils.all_gather(acc_sum.to(device))
        num_vec = misc_utils.all_gather(num_samp.to(device))

        acc_sum = acc_sum.reshape((world_size, intervals))
        acc_sum = acc_sum.sum(dim=0)
        num_samp = num_vec.sum()
    acc_avg = (acc_sum / num_samp).cpu().numpy()  # [intervals]

    print(
        f"[eval mode] input_ids: {input_ids.shape}, acc_mat: {acc_mat.shape},\ngeneration acc:\n{acc_avg}"
    )
    train_cfg.pretrain_mlm.params.umr_clip = default_umr_clip
    print(
        f"reset train_cfg.pretrain_mlm.params.umr_clip to default: {train_cfg.pretrain_mlm.params.umr_clip}"
    )
    model.train()
    return ",".join(acc_avg.round(4).astype(str))


def eval_gen_per_batch(
    model,
    cfg: GenerationConfig,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    inputs_raw_embeds: torch.Tensor,
):
    gen_res, _ = generation_utils.sample_per_batch(
        model,
        cfg,
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_raw_embeds=inputs_raw_embeds,
    )
    acc_vec = generation_utils.cal_gen_acc_batch(
        cfg, input_ids, labels, gen_res
    )  # batch-ver of generation, return vec of [bz]
    return acc_vec


def eval_gen_per_sample(
    model,
    cfg: GenerationConfig,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    inputs_raw_embeds: torch.Tensor,
):
    """
    :param model:
    :param cfg:
    :param input_ids: [bz, seq, next_n]
    :param attention_mask: [bz, seq]
    :param labels: [bz, seq, next_n]
    :param inputs_raw_embeds: [bz, seq, dim] or None
    :return:
    """
    acc_vec = []
    for i in range(len(input_ids)):  # iterate each sample in batch
        # print(f"[{datetime.now()}] i: {i}")
        x = input_ids[i]  # [batch_seq, next_n_token]
        mask = attention_mask[i]  # [batch_seq]
        l = labels[i]  # [batch_seq, next_n_token]
        raw_embeds = None if inputs_raw_embeds is None else inputs_raw_embeds[i]

        non_pad = mask != cfg.pad_token_id
        x = x[non_pad]  # [seq, next_n_token]
        mask = mask[non_pad].unsqueeze(0)  # [1, seq]
        l = l[non_pad]  # [seq, next_n_token]
        raw_embeds = None if raw_embeds is None else raw_embeds[non_pad]

        res, _ = generation_utils.sample_per_example(
            model, cfg, input_ids=x, attention_mask=mask, inputs_raw_embeds=raw_embeds
        )  # res -> [seq*next_n_token]
        acc = generation_utils.cal_gen_acc_per_sample(cfg, x, l, res)
        acc_vec.append(acc)
    acc_vec = torch.hstack(acc_vec)
    return acc_vec


def eval_pt_gen_only(
    model: PreTrainedModel,
    cfg: Config,
    collator_cls,  # src.data.collator.DataCollatorForGST
    tokenizer_cls,  # src.data.tokenizer.GSTTokenizer|StackedGSTTokenizer
    tokenizer_config: Dict,
    pt_sampler: loader_utils.PTSamplerConfig,
    train_dataset: Dataset,
):
    train_cfg = cfg.training
    output_dir = train_cfg.output_dir
    print("\n\nEval-only mode: skip training!!!" * 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set-up valid/test loader and evaluate before training
    valid_loader, valgen_loader = loader_utils.initialize_pt_valid_loader(
        train_dataset, cfg, pt_sampler, tokenizer_config, tokenizer_cls, collator_cls
    )
    ls_ckp = misc_utils.get_all_ckps(output_dir)
    pprint(ls_ckp)
    ls_result = ["ckp,loss,gen_acc1,gen_acc2,gen_acc3,gen_acc4,gen_acc5\n"]
    for ckp in ls_ckp:
        print("\n\n\n")
        if train_cfg.use_deepspeed:
            misc_utils.load_ds_ckp(ckp, model, strict=True)
        else:
            fn_model = os.path.join(output_dir, misc_utils.MODEL_NAME)
            model.load_state_dict(torch.load(fn_model), strict=True)
        model.to(device)
        valid_acc, val_triplet = evaluate(
            model, valid_loader, "valid", train_cfg.do_valid
        )
        valid_gen_acc = evaluate_generation(
            model,
            valgen_loader,
            "valid",
            train_cfg.do_valid and train_cfg.do_generation,
            cfg,
        )
        ls_result.append(f"{ckp.split('/')[-1]},{valid_acc},{valid_gen_acc}\n")

        fn_result = f"{output_dir}/result-gen-{cfg.generation.alg}.csv"
        with open(fn_result, "w") as fp:
            fp.writelines(ls_result)
        print(f"Generation metrics saved in {fn_result}!")
    OmegaConf.save(
        cfg,
        f=os.path.join(
            output_dir,
            f"config-eval-only-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.yaml",
        ),
    )


def log_training_stats(
    training: TrainingConfig,
    train_stats: TrainingStats,
    opt_stats: OptimizingStats,
    prof=None,
    tb_writer=None,
):
    train_stats.cal_speed(training.batch_size)
    train_stats.print_stats()

    # Reduce SUM to get the loss from all the GPUs to RANK=0
    # refer: https://github.com/microsoft/DeepSpeed/discussions/2377#discussioncomment-3765282
    if training.distributed.world_size > 1:
        dist.reduce(train_stats.loss, 0)
        train_stats.loss = train_stats.loss / training.distributed.world_size
    curr_lr = (
        opt_stats.lr_scheduler.get_lr()
        if opt_stats.lr_scheduler is not None
        else [training.lr]
    )
    # flops calculation
    if (not train_stats.use_deepspeed) and (
        train_stats.j % training.schedule.steps_per_saving == 0
    ):
        flops = prof.get_total_flops()
        macs = prof.get_total_macs()
    else:
        flops, macs = 0, 0

    # logging
    train_stats.ls_log.append(
        f"{train_stats.ckp},{curr_lr[0]},{train_stats.i},{train_stats.j},{train_stats.aux_loss},{train_stats.main_loss},{train_stats.loss},{flops},{macs}\n"
    )

    tb_writer.add_scalar(
        "loss", train_stats.loss.item(), train_stats.j
    ) if tb_writer is not None else None


def log_ft_training_stats(
    train_cfg: TrainingConfig, train_stats: TrainingStats, tb_writer=None
):
    train_stats.cal_speed(train_cfg.batch_size)
    train_stats.print_stats()

    # Reduce SUM to get the loss from all the GPUs to RANK=0
    # refer: https://github.com/microsoft/DeepSpeed/discussions/2377#discussioncomment-3765282
    if train_cfg.distributed.world_size > 1:
        dist.reduce(train_stats.loss, 0)
        train_stats.loss = train_stats.loss / train_cfg.distributed.world_size

    # logging
    train_stats.ls_loss.append(
        f"{train_stats.epoch},{train_stats.i},{train_stats.j},{train_stats.aux_loss},{train_stats.main_loss},{train_stats.loss}\n"
    )

    tb_writer.add_scalar(
        "loss", train_stats.loss.item(), train_stats.j
    ) if tb_writer is not None else None


def log_dump_training_stats(
    model: PreTrainedModel,
    cfg: Config,
    train_stats: TrainingStats,
    opt_stats: OptimizingStats,
    loader_stats: LoaderStats,
    ema_stats: EMAStats,
    tb_writer=None,
):
    training = cfg.training
    train_stats.ckp += 1
    train_stats.print_on_saving_ckp(
        training.batch_size, training.distributed.world_size
    )
    misc_utils.save_ckp(
        training.output_dir,
        model,
        train_stats.ckp,
        train_stats.use_deepspeed,
        optimizer=opt_stats.optimizer,
        lr_scheduler=opt_stats.lr_scheduler,
    )

    valid_acc, val_triplet = evaluate(
        model, loader_stats.valid_loader, "valid", training.do_valid
    )
    valid_gen_acc = evaluate_generation(
        model,
        loader_stats.valgen_loader,
        "valid",
        training.do_valid and training.do_generation,
        cfg,
    )
    test_acc, test_triplet = evaluate(
        model, loader_stats.test_loader, "test", training.do_test
    )
    test_gen_acc = evaluate_generation(
        model,
        loader_stats.testgen_loader,
        "test",
        training.do_test and training.do_generation,
        cfg,
    )

    do_ema_eval = training.do_test and ema_stats.model_ema is not None
    ema_acc, ema_triplet = evaluate(
        ema_stats.model_ema, loader_stats.test_loader, "EMA test", do_ema_eval
    )
    if do_ema_eval:
        (
            ema_stats.ema_best_flag,
            ema_stats.ema_best_res,
        ) = metrics_utils.compare_metrics_res({"loss": ema_acc}, ema_stats.ema_best_res)
    train_stats.ls_result.append(
        f"{train_stats.ckp},{train_stats.j},{valid_acc},{test_acc},{ema_acc},{valid_gen_acc},{test_gen_acc}\n"
    )

    if int(os.environ.get("RANK", 0)) == 0:
        # save ckp and logs
        misc_utils.save_all(
            training.output_dir,
            model,
            train_stats.ckp,
            save_model=False,
            ls_log=train_stats.ls_log,
            ls_result=train_stats.ls_result,
            val_dict=val_triplet,
            test_dict=test_triplet,
        )
        ema_stats.save_ema_ckp(training.output_dir)
    print(
        f"[{datetime.now()}]valid/test/EMA eval results: {valid_acc}, {test_acc}, {ema_acc}\nvalid_gen_acc: {valid_gen_acc}\ntest_gen_acc: {test_gen_acc}"
    )
    print(
        f"[{datetime.now()}][input_id] shape: {train_stats.inputs_shape}\n"
        f"    [inputs_raw_embeds(sliced)]:\n{train_stats.sliced_raw_embeds}\n"
    )
    if tb_writer is not None:
        # Log histograms of model parameters
        for name, param in model.named_parameters():
            tb_writer.add_histogram(name, param, train_stats.ckp)


def log_dump_ft_training_stats(
    model: PreTrainedModel,
    cfg: Config,
    ft_sampler: loader_utils.FTSamplerConfig,
    train_stats: TrainingStats,
    opt_stats: OptimizingStats,
    loader_stats: LoaderStats,
    ema_stats: EMAStats,
    tb_writer=None,
):
    train_cfg = cfg.training
    world_size = train_cfg.distributed.world_size
    output_dir = train_cfg.output_dir
    epoch = train_stats.epoch
    i, j = train_stats.i, train_stats.j

    # 0. Save model ckp
    print(
        f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Processing ..."
    )
    if not train_cfg.ft_eval.eval_only:
        print("Saving ckp ...")
        misc_utils.save_ckp(
            train_cfg.output_dir,
            model,
            train_stats.epoch,
            train_stats.use_deepspeed,
            optimizer=opt_stats.optimizer,
            lr_scheduler=opt_stats.lr_scheduler,
        )

    # 1. EVAL on train data
    k_samplers = train_cfg.ft_eval.k_samplers
    print(
        f"[{datetime.now()}] Evaluate on partial train data {k_samplers * world_size} -> {k_samplers}!"
    )
    tr_loss, tr_cls_metrics, tr_ogb_eval_res, tr_triplet = (
        ft_evaluate(model, loader_stats.train_loader_for_eval, cfg, "train")
        if loader_stats.train_loader_for_eval
        else (0, None, None, None)
    )

    # 2. EVAL on valid data
    valid_cnt = ft_sampler.valid.cnt
    valid_sampler = ft_sampler.valid.sampler
    print(
        f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Evaluate on full valid data "
        f"{valid_cnt} -> {len(valid_sampler) if valid_sampler else valid_cnt // world_size}!"
    )
    val_loss, val_cls_metrics, val_ogb_eval_res, val_triplet = ft_evaluate(
        model, loader_stats.valid_loader, cfg, "valid"
    )
    # 2.1 EVAL on valid data with EMA
    val_ogb_eval_res_ema = None
    if ema_stats.model_ema is not None:
        print(
            f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Evaluate on full valid data with "
            f"EMA {valid_cnt} -> {len(valid_sampler) if valid_sampler else valid_cnt // world_size}!"
        )
        (
            val_loss_ema,
            val_cls_metrics_ema,
            val_ogb_eval_res_ema,
            val_triplet_ema,
        ) = ft_evaluate(ema_stats.model_ema, loader_stats.valid_loader, cfg, "valid")
        (
            ema_stats.ema_best_flag,
            ema_stats.ema_best_res,
        ) = metrics_utils.compare_metrics_res(
            val_ogb_eval_res_ema, ema_stats.ema_best_res
        )

    # 3. EVAL on test data -> use EMA if available
    test_cnt = ft_sampler.test.cnt
    test_sampler = ft_sampler.test.sampler
    print(
        f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Evaluate on full test data {test_cnt}"
        f" -> {len(test_sampler) if test_sampler else test_cnt // world_size}!"
    )
    model_for_test = model
    if ema_stats.model_ema is not None:
        model_for_test = ema_stats.model_ema
        print(f"Using model-ema on test data")
    test_loss, test_cls_metrics, test_ogb_eval_res, test_triplet = ft_evaluate(
        model_for_test, loader_stats.test_loader, cfg, "test"
    )

    # 4. Print all eval results
    print(
        f"[{datetime.now()}][epoch {epoch}][local {i}][global {j}] train_loss: {tr_loss}, "
        f"valid_loss: {val_loss}, test_loss: {test_loss}, {test_cls_metrics.results_in_details()},\n"
        f"train ogb_eval: {tr_ogb_eval_res}, valid ogb_eval: {val_ogb_eval_res}, "
        f"EMA valid ogb_eval: {val_ogb_eval_res_ema}, test ogb_eval: {test_ogb_eval_res}"
        f"\nOutput dir: {output_dir}"
    )

    # 5. Formulate all eval results
    train_stats.ls_log.append(
        f"{epoch},{i},{j},{tr_loss},{val_loss},{test_loss},"
        f"{','.join(val_cls_metrics.results_in_str_tuple())},"
        f"{format_ogb_output_for_csv(val_ogb_eval_res)},"
        f"{','.join(test_cls_metrics.results_in_str_tuple())},"
        f"{format_ogb_output_for_csv(test_ogb_eval_res)}\n"
    )
    curr_lr = (
        opt_stats.lr_scheduler.get_lr()
        if opt_stats.lr_scheduler is not None
        else [train_cfg.lr]
    )
    train_stats.ls_result.append(
        f"{epoch},{j},{curr_lr},{curr_lr[0]},"
        f"{format_ogb_output_for_csv(tr_ogb_eval_res)},"
        f"{format_ogb_output_for_csv(val_ogb_eval_res)},"
        f"{format_ogb_output_for_csv(test_ogb_eval_res)},"
        f"{format_ogb_output_for_csv(val_ogb_eval_res_ema)}\n"
    )

    # 6. Save all eval results
    if int(os.environ.get("RANK", 0)) == 0:
        eval_only = train_cfg.ft_eval.eval_only
        save_pred = train_cfg.ft_eval.save_pred
        misc_utils.save_all(
            output_dir,
            model,
            epoch,
            save_model=False,
            ls_log=train_stats.ls_log if not eval_only else None,
            ls_result=train_stats.ls_result if not eval_only else None,
            ls_loss=train_stats.ls_loss if not eval_only else None,
            tr_dict=tr_triplet if save_pred else None,
            val_dict=val_triplet if save_pred else None,
            test_dict=test_triplet if save_pred else None,
        )
        ema_stats.save_ema_ckp(train_cfg.output_dir)

    # 7. Infer logits/hidden states and etc. of test data
    test_cnt = ft_sampler.test.cnt
    test_sampler = ft_sampler.test.sampler
    print(
        f"[{datetime.now()}][end of epoch {epoch}][local {i}][global {j}] Infer on full test data {test_cnt}"
        f" -> {len(test_sampler) if test_sampler else test_cnt // world_size}!"
    )
    model_for_test = model
    if ema_stats.model_ema is not None:
        model_for_test = ema_stats.model_ema
        print(f"Using model-ema on test data")
    idx, logits, hidden_states = ft_infer(
        model_for_test, loader_stats.test_loader, cfg, "test"
    )
    misc_utils.dump_infer_results(
        output_dir, epoch, idx=idx, logits=logits, hidden_states=hidden_states
    )

    # 8. Log params if needed
    if tb_writer is not None:
        # Log histograms of model parameters
        for name, param in model.named_parameters():
            tb_writer.add_histogram(name, param, epoch)


def dump_ds_conf(output_dir, ds_config):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(ds_config)
    with open(os.path.join(output_dir, "ds_config.json"), "w") as fp:
        json.dump(ds_config, fp, indent=4)
    print(f"[{datetime.now()}] Finish -> Dump to `ds_config.json`")


def _dump_ds_cfg_and_init_tb(
    model,
    use_deepspeed: bool,
    use_tb_writer: bool,
    output_dir: str,
    scheduler_conf: dict = None,
):
    if use_deepspeed:
        tmp_ds_config = copy.deepcopy(model.config)
        if scheduler_conf is not None:
            tmp_ds_config.update(scheduler_conf)
        dump_ds_conf(output_dir, tmp_ds_config)

    tb_writer = None
    if use_tb_writer:
        # note: ONLY worker 0 write summary
        # flush_secs: automatic flush, default 120s
        # max_queue: queue size for storing events, default 10; >10 will flush data once to filesystem
        # os.path.join(output_dir, "summary")   os.environ['SUMMARY_DIR']
        summary_dir = os.environ.get("SUMMARY_DIR", os.path.join(output_dir, "summary"))
        tb_writer = SummaryWriter(log_dir=summary_dir, max_queue=30, flush_secs=120)
        print(f"start logging in dir: {summary_dir}")
    return tb_writer


def pt_dump_cfg_and_init_tb(
    model, use_deepspeed: bool, use_tb_writer: bool, output_dir: str
):
    tb_writer = None
    if int(os.environ.get("RANK", 0)) == 0:
        tb_writer = _dump_ds_cfg_and_init_tb(
            model, use_deepspeed, use_tb_writer, output_dir
        )
    return tb_writer


def ft_dump_cfg_and_init_tb(
    model,
    use_deepspeed: bool,
    use_tb_writer: bool,
    output_dir: str,
    eval_only: bool,
    scheduler_conf: dict,
):
    tb_writer = None
    if (int(os.environ.get("RANK", 0)) == 0) and (not eval_only):
        tb_writer = _dump_ds_cfg_and_init_tb(
            model, use_deepspeed, use_tb_writer, output_dir, scheduler_conf
        )
    return tb_writer


@torch.no_grad()
def evaluate_v0(
    model,
    device,
    loader,
):
    # only evaluate the class of last token
    model.eval()
    # cnt_match = 0
    # cnt_all = 0
    if hasattr(model, "module"):
        pad_token_id = model.module.config.pad_token_id
    else:
        pad_token_id = model.config.pad_token_id
    print("Running test data eval ...")
    ls_idx = []
    ls_inputs = []
    ls_pred = []
    ls_true = []
    acc_metric = Accuracy(task="multiclass", num_classes=1000).to(device)
    for test_data in loader:
        input_ids = test_data["input_ids"].to(device)
        attention_mask = test_data["attention_mask"].to(device)
        position_ids = test_data["position_ids"].to(device)
        labels = test_data["labels"].to(device)
        res = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            # position_ids=position_ids,
        )  # Perform a single forward pass.
        logits = res.head1_logits  # [bz, seq, vocab]
        batch_size = input_ids.shape[0]
        if len(input_ids.shape) == 3:
            in_ = input_ids[:, :, 0]  # [bz, seq, num_feat] -> [bz, seq]
        else:
            in_ = input_ids  # [bz, seq]
        sequence_lengths = (torch.ne(in_, pad_token_id).sum(-1) - 1).to(logits.device)
        idx = (torch.arange(batch_size, device=logits.device), sequence_lengths)
        pooled_logits = logits[idx]  # [bz, vocab]
        pred_labels = torch.argmax(pooled_logits, dim=-1)  # [bz]
        true_labels = labels[idx]  # [bz]
        # cnt_match += (pred_labels.cpu().numpy() == true_labels.cpu().numpy()).sum()
        # cnt_all += batch_size
        acc_metric.update(pred_labels, true_labels)

        ls_idx.append(test_data["idx"])
        ls_inputs.append(in_[idx])
        ls_pred.append(pred_labels)
        ls_true.append(true_labels)
    acc = acc_metric.compute().item()
    # acc = cnt_match / cnt_all
    print(f"[eval mode] input_ids shape: {input_ids.shape}, acc: {acc}")
    dict_ = {
        "idx": torch.cat(ls_idx),
        "input_ids": torch.cat(ls_inputs),
        "pred": torch.cat(ls_pred),
        "true": torch.cat(ls_true),
    }
    return acc, dict_
