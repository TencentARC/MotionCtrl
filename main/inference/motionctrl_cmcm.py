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

sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from sgm.util import default, instantiate_from_config

camera_poses = [
    'test_camera_L',
    'test_camera_D',
    'test_camera_I',
    'test_camera_O',
    'test_camera_R',
    'test_camera_U',
    'test_camera_Round-ZoomIn',
    'test_camera_Round-RI_90',
]

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

def get_RT(pose_dir='', video_frames=14, frame_stride=1, speed=1.0, **kwargs):
    pose_file = [f'{pose_dir}/{pose}.json' for pose in camera_poses]
    pose_sample_num = len(pose_file)

    pose_sample_num = len(pose_file)

    data_list = []
    pose_name = []


    for idx in range(pose_sample_num):
        cur_pose_name = camera_poses[idx].replace('test_camera_', '')
        pose_name.append(cur_pose_name)

        with open(pose_file[idx], 'r') as f:
            pose = json.load(f)
        pose = np.array(pose) # [t, 12]
        
        while frame_stride * video_frames > pose.shape[0]:
            frame_stride -= 1

        pose = pose[::frame_stride]
        if video_frames < 16:
            half = (pose.shape[0] - video_frames) // 2
            pose = pose[half:half+video_frames]
        # pose = pose[:video_frames]
        pose = pose.reshape(-1, 3, 4) # [t, 3, 4]
        # rescale
        pose[:, :, -1] = pose[:, :, -1] * np.array([3, 1, 4]) * speed
        pose = to_relative_RT2(pose)
        
            
        pose = torch.tensor(pose).float() # [t, 12]
        data_list.append(pose)

    # data_list = torch.stack(data_list, dim=0) # [pose_sample_num, t, 12]
    return data_list, pose_name

def sample(
    input_path: str = "examples/camera_poses",  # Can either be image file or folder with image files
    ckpt: str = "checkpoints/motionctrl_svd.ckpt",
    config: str = None,
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 1,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    save_fps: int = 10,
    resize: Optional[bool] = False,
    pose_dir: str = '',
    sample_num: int = 1,
    height: int = 576,
    width: int = 1024,
    transform: Optional[bool] = False,
    save_images: Optional[bool] = False,
    speed: float = 1.0,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    assert (version == "svd"), "Only SVD is supported for now."
    num_frames = default(num_frames, 14)
    num_steps = default(num_steps, 25)
    output_folder = default(output_folder, "outputs/motionctrl_svd/")
    model_config = default(config, "configs/inference/config_motionctrl_cmcm.yaml")

    model, filter = load_model(
        model_config,
        ckpt,
        device,
        num_frames,
        num_steps,
    )
    torch.manual_seed(seed)

    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError
    
    if transform:
        spatial_transform = Compose([
            Resize(size=width),
            CenterCrop(size=(height, width)),
        ])
    
    # get camera poses
    RTs, pose_name = get_RT(pose_dir=pose_dir, video_frames=num_frames, frame_stride=1, speed=speed)

    print(f'loaded {len(all_img_paths)} images.')
    os.makedirs(output_folder, exist_ok=True)
    for no, input_img_path in enumerate(all_img_paths):
        
        filepath, fullflname = os.path.split(input_img_path)
        filename, ext = os.path.splitext(fullflname)
        print(f'-sample {no+1}: {filename} ...')

        # RTs = RTs[0:1]
        for RT_idx in range(len(RTs)):
            cur_pose_name = pose_name[RT_idx]
            print(f'--pose: {cur_pose_name} ...')
            RT = RTs[RT_idx]
            RT = RT.unsqueeze(0).repeat(2,1,1)
            RT = RT.to(device)

            with Image.open(input_img_path) as image:
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                if transform:
                    image = spatial_transform(image)
                if resize:
                    image = image.resize((width, height))
                w, h = image.size

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
            if (H, W) != (576, 1024):
                print(
                    "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
                )
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

                    
                    additional_model_inputs["RT"] = RT

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

                    video_path = os.path.join(output_folder, f"{filename}_{cur_pose_name}.mp4")
                    save_results(samples, video_path, fps=save_fps)

                    if save_images:
                        for i in range(sample_num):
                            cur_output_folder = os.path.join(output_folder, f"{filename}", f"{cur_pose_name}", f"{i}")
                            os.makedirs(cur_output_folder, exist_ok=True)
                            for j in range(num_frames):
                                cur_img_path = os.path.join(cur_output_folder, f"{j:06d}.png")
                                torchvision.utils.save_image(samples[i,j], cur_img_path)
    
    print(f'Done! results saved in {output_folder}.')

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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--input", type=str, default=None, help="image path or folder")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=int, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=6, help="control the fps")
    parser.add_argument("--motion", type=int, default=127, help="control the motion magnitude")
    parser.add_argument("--cond_aug", type=float, default=0.02, help="adding noise to input image")
    parser.add_argument("--decoding_t", type=int, default=1, help="frames num to decoding per time")
    parser.add_argument("--resize", action='store_true', default=False, help="resize all input to default resolution")
    parser.add_argument("--sample_num", type=int, default=1, help="frames num to decoding per time")
    parser.add_argument("--pose_dir", type=str, default='', help="checkpoint path")
    parser.add_argument("--height", type=int, default=576, help="frames num to decoding per time")
    parser.add_argument("--width", type=int, default=1024, help="frames num to decoding per time")
    parser.add_argument("--transform", action='store_true', default=False, help="resize all input to specific resolution")
    parser.add_argument("--save_images", action='store_true', default=False, help="save images")
    parser.add_argument("--speed", type=float, default=1.0, help="speed of camera motion")
    return parser


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@MotionCrl+SVD Inference: %s"%now)
    #Fire(sample)
    parser = get_parser()
    args = parser.parse_args()
    sample(input_path=args.input, ckpt=args.ckpt, config=args.config, num_frames=args.frames, num_steps=args.ddim_steps, \
        fps_id=args.fps, motion_bucket_id=args.motion, cond_aug=args.cond_aug, seed=args.seed, \
        decoding_t=args.decoding_t, output_folder=args.savedir, save_fps=args.savefps, resize=args.resize,
        pose_dir=args.pose_dir, sample_num=args.sample_num, height=args.height, width=args.width,
        transform=args.transform, save_images=args.save_images, speed=args.speed)
    
