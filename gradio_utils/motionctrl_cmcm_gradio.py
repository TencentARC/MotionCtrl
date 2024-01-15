import argparse
import datetime
import json
import math
import os
import sys
import time
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor
import tempfile

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from sgm.util import default, instantiate_from_config



def to_relative_RT2(org_pose, keyframe_idx=0, keyframe_zero=False):
        org_pose = org_pose.reshape(-1, 3, 4) # [t, 3, 4]
        R_dst = org_pose[:, :, :3]
        T_dst = org_pose[:, :, 3:]

        R_src = R_dst[keyframe_idx: keyframe_idx+1].repeat(org_pose.shape[0], axis=0) # [t, 3, 3]
        T_src = T_dst[keyframe_idx: keyframe_idx+1].repeat(org_pose.shape[0], axis=0)

        R_src_inv = R_src.transpose(0, 2, 1) # [t, 3, 3]
        
        R_rel = R_dst @ R_src_inv # [t, 3, 3]
        T_rel = T_dst - R_rel@T_src

        RT_rel = np.concatenate([R_rel, T_rel], axis=-1) # [t, 3, 4]
        RT_rel = RT_rel.reshape(-1, 12) # [t, 12]

        if keyframe_zero:
            RT_rel[keyframe_idx] = np.zeros_like(RT_rel[keyframe_idx])

        return RT_rel

def build_model(config, ckpt, device, num_frames, num_steps):
    num_frames = default(num_frames, 14)
    num_steps = default(num_steps, 25)
    model_config = default(config, "configs/inference/config_motionctrl_cmcm.yaml")

    print(f"Loading model from {ckpt}")
    model, filter = load_model(
        model_config,
        ckpt,
        device,
        num_frames,
        num_steps,
    )

    model.eval()

    return model

def motionctrl_sample(
    model,
    image: Image = None,  # Can either be image file or folder with image files
    RT: np.ndarray = None,
    num_frames: Optional[int] = None,
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 1,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    save_fps: int = 10,
    sample_num: int = 1,
    device: str = "cuda",
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    torch.manual_seed(seed)

    w, h = image.size

    # RT: [t, 3, 4]
    # RT = RT.reshape(-1, 3, 4) # [t, 3, 4]
    # adaptive to different spatial ratio
    # base_len = min(w, h) * 0.5
    # K = np.array([[w/base_len, 0, w/base_len],
    #               [0, h/base_len, h/base_len],
    #               [0, 0, 1]])
    # for i in range(RT.shape[0]):
    #     RT[i,:,:] = np.dot(K, RT[i,:,:])

    RT = to_relative_RT2(RT) # [t, 12]
    RT = torch.tensor(RT).float().to(device) # [t, 12]
    RT = RT.unsqueeze(0).repeat(2,1,1)

    if h % 64 != 0 or w % 64 != 0:
        width, height = map(lambda x: x - x % 64, (w, h))
        image = image.resize((width, height))
        print(
            f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
        )

    image = ToTensor()(image)
    image = image * 2.0 - 1.0

    image = image.unsqueeze(0).to(device)
    H, W = image.shape[2:]
    assert image.shape[1] == 3
    F = 8
    C = 4
    shape = (num_frames, C, H // F, W // F)

    if motion_bucket_id > 255:
        print(
            "WARNING: High motion bucket! This may lead to suboptimal performance."
        )

    if fps_id < 5:
        print("WARNING: Small fps value! This may lead to suboptimal performance.")

    if fps_id > 30:
        print("WARNING: Large fps value! This may lead to suboptimal performance.")

    value_dict = {}
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames_without_noise"] = image
    value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)

    with torch.no_grad():
        with torch.autocast(device):
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )

            for k in ["crossattn", "concat"]:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

            

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, num_frames
            ).to(device)
            #additional_model_inputs["image_only_indicator"][:,0] = 1
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

            
            additional_model_inputs["RT"] = RT.clone()

            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model, input, sigma, c, **additional_model_inputs
                )

            results = []
            for j in range(sample_num):
                randn = torch.randn(shape, device=device)
                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0) # [1*t, c, h, w]
                results.append(samples)

            samples = torch.stack(results, dim=0) # [sample_num, t, c, h, w]
            samples = samples.data.cpu()

            video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
            save_results(samples, video_path, fps=save_fps)
    
    return video_path

def save_results(resutls, filename, fps=10):
    video = resutls.permute(1, 0, 2, 3, 4) # [t, sample_num, c, h, w]
    frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(video.shape[1])) for framesheet in video] #[3, 1*h, n*w]
    grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
    # already in [0,1]
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
    torchvision.io.write_video(filename, grid, fps=fps, video_codec='h264', options={'crf': '10'})

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    ckpt: str,
    device: str,
    num_frames: int,
    num_steps: int,
):

    config = OmegaConf.load(config)
    config.model.params.ckpt_path = ckpt
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )

    model = instantiate_from_config(config.model)

    model = model.to(device).eval()    

    filter = None #DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter

