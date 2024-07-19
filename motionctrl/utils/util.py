import importlib
import os
from typing import Union

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from einops import rearrange
from safetensors import safe_open
from tqdm import tqdm

# from motionctrl.utils.convert_from_ckpt import (convert_ldm_clip_checkpoint,
#                                                  convert_ldm_unet_checkpoint,
#                                                  convert_ldm_vae_checkpoint)
# from motionctrl.utils.convert_lora_safetensor_to_diffusers import (
#     convert_lora, convert_motion_lora_ckpt_to_diffusers)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

