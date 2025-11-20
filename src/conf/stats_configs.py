import os
import torch
from datetime import datetime
from typing import Optional, List, Union
from dataclasses import dataclass, field
from torch import FloatTensor, Tensor, Size
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV3
from timm.models import load_checkpoint
from timm.utils.model import unwrap_model, get_state_dict


@dataclass(frozen=True)
class OptimizingStats:
    optimizer: torch.optim.Optimizer
    lr_scheduler: Union[_LRScheduler, ReduceLROnPlateau]
    scaler: Optional[GradScaler] = None


@dataclass
class OdpsStats:
    is_odps_table_ds: bool = False
    steps_per_epoch: Optional[int] = None
    tables: Optional[str] = None


@dataclass
class TrainingStats:
    device: torch.device
    has_embeds_input: bool = False

    use_deepspeed: bool = False

    ls_log: List[str] = None
    ls_result: List[str] = None
    ls_loss: List[str] = None

    tr_dict: dict = None
    val_dictt: dict = None
    test_dict: dict = None

    reset_samples_per_epoch: bool = False
    t_start: datetime = None

    tokens_per_sample: int = 0

    ckp: int = 0
    epoch_start: int = 0
    epoch: int = 0

    i_local: int = 0
    i: int = 0
    j: int = 0

    loss: Optional[FloatTensor] = None
    main_loss: Optional[FloatTensor] = None
    aux_loss: Optional[FloatTensor] = None

    inputs_shape: Optional[Size] = None
    sliced_raw_embeds: Optional[Tensor] = None

    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0

    odps_stats: OdpsStats = field(default_factory=OdpsStats)

    def cal_speed(self, batch_size: int):
        t_interval = (datetime.now() - self.t_start).total_seconds()
        self.samples_per_second = round(
            (self.i - self.i_local) * batch_size / t_interval, 1
        )
        self.tokens_per_second = round(
            (self.i - self.i_local) * batch_size * self.tokens_per_sample / t_interval
        )

    def print_stats(self):
        if self.aux_loss is not None:
            loss_log = f"\n{' ' * 8}loss: {round(self.loss.item(), 6)}, aux_Loss {round(self.aux_loss.item(), 6)}, main/task_Loss {round(self.main_loss.item(), 6)}"
        else:
            loss_log = f"loss: {round(self.loss.item(), 6)}"
        print(
            f"[{datetime.now()}][epoch {self.ckp}][local {self.epoch}: {self.i}][global {self.j}] {self.samples_per_second} samples / {self.tokens_per_second} tokens per sec; {loss_log}"
        )

    def print_on_saving_ckp(self, batch_size, world_size):
        print(
            f"[{datetime.now()}][end of ckp {self.ckp}][local {self.epoch}: {self.i}][global {self.j}] "
            f"Trained with {int(self.j * self.tokens_per_sample * batch_size * world_size)} tokens! Saving ckp and logs!"
        )


@dataclass
class EMAConfig:
    use_ema: bool = False
    ema_file: str = "model_ema.pt"
    ema_file_best: str = "model_ema_best.pt"


@dataclass
class EMAStats:
    model_ema: Optional[torch.nn.Module] = None
    ema_cfg: EMAConfig = field(default_factory=EMAConfig)

    ema_best_flag: bool = False
    ema_best_res: Optional[dict] = None

    def init_ema(
        self, model, ema_module: torch.nn.Module = None, decay: float = 0.9999
    ):
        if self.ema_cfg.use_ema:
            if ema_module:
                self.model_ema = ema_module(model.module, decay)
            else:
                self.model_ema = ModelEmaV3(model.module, decay)

    def ema2device(self, device, use_ema: bool):
        if use_ema:
            self.model_ema.module.to(device=device)
            emb = self.model_ema.module.model.embed_tokens.weight.data
            print(f"[Debug] model-ema embedding_params:\n{emb}\n{emb.shape}")

    def load_ema_ckp(self, output_dir):
        if self.model_ema is not None:
            ema_ckp = os.path.join(output_dir, self.ema_cfg.ema_file)
            load_checkpoint(self.model_ema.module, ema_ckp, use_ema=True)
            print(f"load model_ema ckp from {ema_ckp}")

    def update_ema(self, model, step: int, ft: bool = False):
        if self.model_ema is not None:
            if ft:
                self.model_ema.update(model.module, step=step)
            else:
                self.model_ema.update(model, step=step)

    def save_ema_ckp(self, output_dir):
        if self.model_ema is not None:
            ema_state = get_state_dict(self.model_ema, unwrap_model)
            ema_ckp = os.path.join(output_dir, self.ema_cfg.ema_file)
            torch.save(ema_state, ema_ckp)
            if self.ema_best_flag:
                torch.save(
                    ema_state, os.path.join(output_dir, self.ema_cfg.ema_file_best)
                )


@dataclass(frozen=True, kw_only=True)
class LoaderStats:
    train_loader: Optional[DataLoader] = None

    train_loader_for_eval: Optional[DataLoader] = None
    valid_loader: Optional[DataLoader] = None
    valgen_loader: Optional[DataLoader] = None  # for generation
    test_loader: Optional[DataLoader] = None
    testgen_loader: Optional[DataLoader] = None  # for generation
