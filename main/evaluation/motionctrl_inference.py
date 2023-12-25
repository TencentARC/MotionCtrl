import argparse
import datetime
import glob
import json
import math
import os
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision
## note: decord should be imported after torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from main.evaluation.motionctrl_prompts_camerapose_trajs import (
    both_prompt_camerapose_traj, cmcm_prompt_camerapose, omom_prompt_traj)
from utils.utils import instantiate_from_config

DEFAULT_NEGATIVE_PROMPT = 'blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, '\
                          'sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, '\
                          'disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, '\
                          'floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation'

post_prompt = 'Ultra-detail, masterpiece, best quality, cinematic lighting, 8k uhd, dslr, soft lighting, film grain, Fujifilm XT3'


def load_model_checkpoint(model, ckpt, adapter_ckpt=None):
    if adapter_ckpt:
        ## main model
        state_dict = torch.load(ckpt, map_location="cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
            result = model.load_state_dict(state_dict, strict=False)
        else:       
            # deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]]=state_dict['module'][key]
            result = model.load_state_dict(new_pl_sd, strict=False)
        print(result)
        print('>>> model checkpoint loaded.')
        ## adapter
        state_dict = torch.load(adapter_ckpt, map_location="cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        model.adapter.load_state_dict(state_dict, strict=True)
        print('>>> adapter checkpoint loaded.')
    else:
        state_dict = torch.load(ckpt, map_location="cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
        else:       
            # deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]]=state_dict['module'][key]
            model.load_state_dict(new_pl_sd)
        
        print('>>> model checkpoint loaded.')
    return model

def load_trajs(cond_dir, trajs):
    traj_files = [f'{cond_dir}/trajectories/{traj}.npy' for traj in trajs]

    data_list = []
    traj_name = []

    for idx in range(len(traj_files)):
        traj_name.append(traj_files[idx].split('/')[-1].split('.')[0])
        data_list.append(torch.tensor(np.load(traj_files[idx])).permute(3, 0, 1, 2).float()) # [t,h,w,c] -> [c,t,h,w]
    
    return data_list, traj_name

def load_camera_pose(cond_dir, camera_poses):
    
    pose_file = [f'{cond_dir}/camera_poses/{pose}.json' for pose in camera_poses]
    pose_sample_num = len(pose_file)

    data_list = []
    pose_name = []

    for idx in range(pose_sample_num):
        cur_pose_name = camera_poses[idx].replace('test_camera_', '')
        pose_name.append(cur_pose_name)

        with open(pose_file[idx], 'r') as f:
            pose = json.load(f)
        pose = np.array(pose) # [t, 12]
        pose = torch.tensor(pose).float() # [t, 12]
        data_list.append(pose)

    return data_list, pose_name

def save_results(samples, filename, savedir, fps=10):
    ## save prompt

    ## save video
    videos = [samples]
    savedirs = [savedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        path = os.path.join(savedirs[idx], "%s.mp4"%filename)
        torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})

def motionctrl_sample(
        model, 
        prompts, 
        noise_shape,
        camera_poses=None, 
        trajs=None,
        n_samples=1,
        unconditional_guidance_scale=1.0,
        unconditional_guidance_scale_temporal=None,
        ddim_steps=50,
        ddim_eta=1.,
        **kwargs):
    
    ddim_sampler = DDIMSampler(model)
    batch_size = noise_shape[0]
    ## get condition embeddings (support single prompt only)
    if isinstance(prompts, str):
        prompts = [prompts]

    for i in range(len(prompts)):
        prompts[i] = f'{prompts[i]}, {post_prompt}'

    cond = model.get_learned_conditioning(prompts)
    if camera_poses is not None:
        RT = camera_poses[..., None]
    else:
        RT = None

    if trajs is not None:
        traj_features = model.get_traj_features(trajs)
    else:
        traj_features = None

    if unconditional_guidance_scale != 1.0:
        # prompts = batch_size * [""]
        prompts = batch_size * [DEFAULT_NEGATIVE_PROMPT]
        uc = model.get_learned_conditioning(prompts)
        if traj_features is not None:
            un_motion = model.get_traj_features(torch.zeros_like(trajs))
        else:
            un_motion = None
        uc = {"features_adapter": un_motion, "uc": uc}
    else:
        uc = None

    batch_variants = []
    for _ in range(n_samples):
        if ddim_sampler is not None:
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=unconditional_guidance_scale_temporal,
                                            features_adapter=traj_features,
                                            pose_emb=RT,
                                            **kwargs
                                            )        
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)

def run_inference(args, gpu_num, gpu_no):
    ## model config
    config = OmegaConf.load(args.base)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint {args.ckpt_path} Not Found!"
    print(f"Loading checkpoint from {args.ckpt_path}")
    model = load_model_checkpoint(model, args.ckpt_path, args.adapter_ckpt)
    model.eval()

    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.channels
    frames = model.temporal_length
    noise_shape = [args.bs, channels, frames, h, w]

    savedir = os.path.join(args.savedir, "samples")
    os.makedirs(savedir, exist_ok=True)

    if args.condtype == 'camera_motion':
        prompt_list = cmcm_prompt_camerapose['prompts']
        camera_pose_list, pose_name = load_camera_pose(args.cond_dir, cmcm_prompt_camerapose['camera_poses'])
        traj_list = None
        save_name_list = []
        for i in range(len(pose_name)):
            save_name_list.append(f"{pose_name[i]}__{prompt_list[i].replace(' ', '_').replace(',', '')}")
    elif args.condtype == 'object_motion':
        prompt_list = omom_prompt_traj['prompts']
        traj_list, traj_name = load_trajs(args.cond_dir, omom_prompt_traj['trajs'])
        camera_pose_list = None
        save_name_list = []
        for i in range(len(traj_name)):
            save_name_list.append(f"{traj_name[i]}__{prompt_list[i].replace(' ', '_').replace(',', '')}")
    elif args.condtype == 'both':
        prompt_list = both_prompt_camerapose_traj['prompts']
        camera_pose_list, pose_name = load_camera_pose(args.cond_dir, both_prompt_camerapose_traj['camera_poses'])
        traj_list, traj_name = load_trajs(args.cond_dir, both_prompt_camerapose_traj['trajs'])
        save_name_list = []
        for i in range(len(pose_name)):
            save_name_list.append(f"{pose_name[i]}__{traj_name[i]}__{prompt_list[i].replace(' ', '_').replace(',', '')}")
    
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
    #indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    prompt_list_rank = [prompt_list[i] for i in indices]
    camera_pose_list_rank = None if camera_pose_list is None else [camera_pose_list[i] for i in indices]
    traj_list_rank = None if traj_list is None else [traj_list[i] for i in indices]
    save_name_list_rank = [save_name_list[i] for i in indices]
    
    start = time.time() 
    for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
        prompts = prompt_list_rank[indice:indice+args.bs]
        camera_poses = None if camera_pose_list_rank is None else camera_pose_list_rank[indice:indice+args.bs]
        trajs = None if traj_list_rank is None else traj_list_rank[indice:indice+args.bs]
        save_name = save_name_list_rank[indice:indice+args.bs]
        print(f'Processing {save_name}')

        if camera_poses is not None:
            camera_poses = torch.stack(camera_poses, dim=0).to("cuda")
        if trajs is not None:
            trajs = torch.stack(trajs, dim=0).to("cuda")

        batch_samples = motionctrl_sample(
            model, 
            prompts, 
            noise_shape,
            camera_poses=camera_poses,
            trajs=trajs,
            n_samples=args.n_samples,
            unconditional_guidance_scale=args.unconditional_guidance_scale,
            unconditional_guidance_scale_temporal=args.unconditional_guidance_scale_temporal,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
            cond_T = args.cond_T,
        )
        
        ## save each example individually
        for nn, samples in enumerate(batch_samples):
            ## samples : [n_samples,c,t,h,w]
            prompt = prompts[nn]
            name = save_name[nn]
            if len(name) > 90:
                name = name[:90]
            filename = f'{name}_{idx*args.bs+nn:04d}_randk{gpu_no}'
            
            save_results(samples, filename, savedir, fps=10)
            if args.save_imgs:
                parts = save_name[nn].split('__')
                if len(parts) == 2:
                    cond_name = parts[0]
                    prname = prompts[nn].replace(' ', '_').replace(',', '')
                    cur_outdir = os.path.join(savedir, cond_name, prname)
                elif len(parts) == 3:
                    poname, trajname, _ = save_name[nn].split('__')
                    prname = prompts[nn].replace(' ', '_').replace(',', '')
                    cur_outdir = os.path.join(savedir, poname, trajname, prname)
                else:
                    raise NotImplementedError
                os.makedirs(cur_outdir, exist_ok=True)
                save_images(samples, cur_outdir)
            if nn % 100 == 0:
                print(f'Finish {nn}/{len(batch_samples)}')

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")

def save_images(samples, savedir):
    ## samples : [n_samples,c,t,h,w]
    n_samples, c, t, h, w = samples.shape
    samples = torch.clamp(samples, -1.0, 1.0)
    samples = (samples + 1.0) / 2.0
    samples = (samples * 255).detach().cpu().numpy().astype(np.uint8)
    for i in range(n_samples):
        cur_outdir = os.path.join(savedir, f'{i}/images')
        os.makedirs(cur_outdir, exist_ok=True)

        for j in range(t):
            img = samples[i,:,j,:,:]
            img = np.transpose(img, (1,2,0))
            img = img[:,:,::-1] # BGR to RGB
            path = os.path.join(cur_outdir, f'{j:04d}.png')
            cv2.imwrite(path, img)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--adapter_ckpt", type=str, default=None, help="adapter checkpoint path")
    parser.add_argument("--base", type=str, help="config (yaml) path")
    parser.add_argument("--condtype", default='frame', type=str, help="conditon type: {frame, depth, adapter}")
    parser.add_argument("--prompt_dir", type=str, default=None, help="a data dir containing videos and prompts")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--cond_T", default=800, type=int, help="Steps smaller than cond_T will not contain condition")
    parser.add_argument("--save_imgs", action='store_true', help="save condition")
    parser.add_argument("--cond_dir", type=str, default=None, help="condition dir")
    
    return parser


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM cond-Inference: %s"%now)
    parser = get_parser()
    args, unkown = parser.parse_known_args()
    # args = parser.parse_args()

    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)