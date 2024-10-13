import torch
from timm.utils import ModelEmaV3


# below copied from timm.utils.model_ema.py::ModelEmaV3::apply_update_ and modified
# MAIN modification is to convert `model_v`'s dtype to be the same as ema_v
# Usually, `ema_v` shall be fp32, and in DeepSpeed, `model_v` is usually fp16
# so, converting to fp32 and the do EMA => If we can find way to direct access fp32 state dict in DS, would be BEST!
# If using fp16 to do EMA in DeepSpeed, the ema weights won't be updated!
def apply_update_(self, model, decay: float):
    # interpolate parameters and buffers
    if self.foreach:
        ema_lerp_values = []
        model_lerp_values = []
        for ema_v, model_v in zip(
            self.module.state_dict().values(), model.state_dict().values()
        ):
            if ema_v.is_floating_point():
                ema_lerp_values.append(ema_v)
                model_lerp_values.append(model_v.to(dtype=ema_v.dtype))
            else:
                ema_v.copy_(model_v)

        if hasattr(torch, "_foreach_lerp_"):
            torch._foreach_lerp_(ema_lerp_values, model_lerp_values, weight=1.0 - decay)
        else:
            torch._foreach_mul_(ema_lerp_values, scalar=decay)
            torch._foreach_add_(ema_lerp_values, model_lerp_values, alpha=1.0 - decay)
    else:
        for ema_v, model_v in zip(
            self.module.state_dict().values(), model.state_dict().values()
        ):
            if ema_v.is_floating_point():
                ema_v.lerp_(
                    model_v.to(device=self.device, dtype=ema_v.dtype),
                    weight=1.0 - decay,
                )
            else:
                ema_v.copy_(model_v.to(device=self.device))


ModelEmaV3.apply_update_ = apply_update_
