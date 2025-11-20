import torch
from typing import Dict
from transformers.modeling_utils import PreTrainedModel
from ..conf import TrainingStats, OptimizingStats, TrainingConfig, FinetuningHeadConfig


def batch_training(
    data: Dict[str, torch.Tensor],
    model: PreTrainedModel,
    train_cfg: TrainingConfig,
    train_stats: TrainingStats,
    opt_stats: OptimizingStats,
):
    optim_cfg = train_cfg.optimizer
    device = train_stats.device

    input_ids = data["input_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)
    position_ids = data["position_ids"].to(device)
    labels = data["labels"].to(device)
    inputs_raw_embeds = None
    if train_stats.has_embeds_input:
        inputs_raw_embeds = data["embed"].to(device)
    sample_wgt = None
    if "wgt" in data:
        sample_wgt = data["wgt"].to(device)

    loss = main_loss = aux_loss = None
    if train_stats.use_deepspeed:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            inputs_raw_embeds=inputs_raw_embeds,
            # position_ids=position_ids,
            sample_wgt=sample_wgt,
        )  # Perform a single forward pass.
        main_loss = output.head1_loss
        aux_loss = output.head2_loss
        if aux_loss is not None:
            loss = main_loss + aux_loss
        else:
            loss = main_loss
        model.backward(loss)  # Derive gradients.
        model.step()
    else:
        assert (
            optim_cfg.gradient_accumulation_steps == 1
        ), "https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation"
        opt_stats.optimizer.zero_grad()  # Clear gradients.
        # https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
        # Enables autocasting for the forward pass (model + loss)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                inputs_raw_embeds=inputs_raw_embeds,
                # position_ids=position_ids,
                sample_wgt=sample_wgt,
            )  # Perform a single forward pass.
            main_loss = output.head1_loss
            aux_loss = output.head2_loss
            if aux_loss is not None:
                loss = main_loss + aux_loss
            else:
                loss = main_loss
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        opt_stats.scaler.scale(loss).backward()
        if optim_cfg.max_grad_norm > 0:
            # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            # Unscales the gradients of optimizer's assigned params in-place
            opt_stats.scaler.unscale_(opt_stats.optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.max_grad_norm)

        # IF not unscaled, scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        opt_stats.scaler.step(opt_stats.optimizer)

        # Updates the scale for next iteration.
        opt_stats.scaler.update()
        opt_stats.lr_scheduler.step()
    train_stats.loss = loss
    train_stats.main_loss = main_loss
    train_stats.aux_loss = aux_loss

    # records for printing out and inspecting
    train_stats.inputs_shape = input_ids.shape
    train_stats.sliced_raw_embeds = (
        inputs_raw_embeds[:2, :8] if inputs_raw_embeds is not None else None
    )


def ft_batch_training(
    data: Dict[str, torch.Tensor],
    model: PreTrainedModel,
    fthead_cfg: FinetuningHeadConfig,
    train_cfg: TrainingConfig,
    train_stats: TrainingStats,
    opt_stats: OptimizingStats,
):
    optim_cfg = train_cfg.optimizer
    device = train_stats.device
    i = train_stats.i
    autocast_type = torch.float16

    print(
        f"[sample idx top 10][local i:{i}]{data['idx'][:10]} {data['input_ids'].shape}\n"
        f"inputs keys: {data.keys()}"
    ) if i == 0 else None
    input_ids = data["input_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)
    position_ids = data["position_ids"].to(device)
    labels = data["labels"].to(device)
    task_labels = data[f"{fthead_cfg.task_type}_labels"].to(device)
    task_labels = (
        task_labels.float()
        if fthead_cfg.problem_type == "multi_label_classification"
        else task_labels
    )
    cls_idx = data["cls_idx"].to(device) if "cls_idx" in data else None
    inputs_raw_embeds = None
    if train_stats.has_embeds_input:
        inputs_raw_embeds = data["embed"].to(device)
    sample_wgt = None
    if "wgt" in data:
        sample_wgt = data["wgt"].to(device)
    if "noise" in data:
        labels = data["noise"].to(device)

    if train_stats.use_deepspeed:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pretrain_labels=labels if train_cfg.finetune.use_aux else None,
            task_labels=task_labels,
            cls_idx=cls_idx,
            inputs_raw_embeds=inputs_raw_embeds,
            sample_wgt=sample_wgt,
            position_ids=position_ids,
        )  # Perform a single forward pass.
        aux_loss = output.pretrain_loss
        task_loss = output.task_loss
        if aux_loss is not None:
            # loss = (
            #     aux_loss.float() * aux_ratio
            #     + task_loss.float() * task_ratio
            # )
            loss = aux_loss.float() + task_loss.float()
        else:
            loss = task_loss.float()
        model.backward(loss)  # Derive gradients.
        model.step()
    else:
        assert (
            optim_cfg.gradient_accumulation_steps == 1
        ), "https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation"
        opt_stats.optimizer.zero_grad()  # Clear gradients.
        # https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
        # Enables autocasting for the forward pass (model + loss)
        with torch.autocast(device_type=device.type, dtype=autocast_type):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pretrain_labels=labels if train_cfg.finetune.use_aux else None,
                task_labels=task_labels,
                cls_idx=cls_idx,
                inputs_raw_embeds=inputs_raw_embeds,
                sample_wgt=sample_wgt,
                position_ids=position_ids,
            )  # Perform a single forward pass.
            aux_loss = output.pretrain_loss
            task_loss = output.task_loss
            if aux_loss is not None:
                # loss = aux_loss * aux_ratio + task_loss * task_ratio
                loss = aux_loss + task_loss
            else:
                loss = task_loss
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        opt_stats.scaler.scale(loss).backward()
        if optim_cfg.max_grad_norm > 0:
            # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            # Unscales the gradients of optimizer's assigned params in-place
            opt_stats.scaler.unscale_(opt_stats.optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.max_grad_norm)

        # IF not unscaled, scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        opt_stats.scaler.step(opt_stats.optimizer)

        # Updates the scale for next iteration.
        opt_stats.scaler.update()
        opt_stats.lr_scheduler.step()

    train_stats.loss = loss
    train_stats.main_loss = task_loss
    train_stats.aux_loss = aux_loss
