from typing import Dict


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
