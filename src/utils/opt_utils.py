import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from ..conf import TrainingConfig, OptimizingStats


def initialize_optimizer(model, model_parameters, training: TrainingConfig, loss_utils):
    sched_cfg = training.schedule
    optim_cfg = training.optimizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = DDP(model.to(device), find_unused_parameters=False)
    except ValueError:
        print("In local test setting!!!\n" * 5)
    # if optimizer is None:
    #     cus_opt = StableAdamW if use_stable else torch.optim.AdamW
    optimizer = torch.optim.AdamW(
        model_parameters,
        lr=optim_cfg.lr,
        betas=optim_cfg.betas,
        eps=optim_cfg.eps,
        weight_decay=optim_cfg.weight_decay,
    )
    lr_scheduler_generator, _ = loss_utils.set_py_scheduler(
        "OneCycleLR",
        {"scheduler": {"params": {}}},
        max_lr=optim_cfg.lr,
        min_lr=optim_cfg.min_lr,
        total_steps=sched_cfg.total_num_steps + 1,
        pct_start=sched_cfg.warmup_num_steps / sched_cfg.total_num_steps,
        last_step_index=-1,
    )  # total_num_steps+1 to avoid error of lr_scheduler.step() in last step
    lr_scheduler = lr_scheduler_generator(optimizer)
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()
    return model, OptimizingStats(optimizer, lr_scheduler, scaler)
