import argparse
import datetime
import inspect
import os, sys
from omegaconf import OmegaConf
import json

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob, math
from pathlib import Path
from PIL import Image
import numpy as np
from motionctrl.modified_modules import (
    Adapted_TemporalTransformerBlock_forward, unet3d_forward,
    Adapted_CrossAttnDownBlock3D_forward, Adapted_DownBlock3D_forward)

from motionctrl.adapter import Adapter
from motionctrl.utils.util import instantiate_from_config
from motionctrl.util import get_traj_features, get_batch_motion, get_opt_from_video, vis_opt_flow


@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}" #-{time_str}"

    config  = OmegaConf.load(args.config)

    samples = []

    # create validation pipeline
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)
        savedir = f"{savedir}_H{model_config.H}_W{model_config.W}"

        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).cuda()

        # load controlnet model
        controlnet = controlnet_images = None
        if model_config.get("controlnet_path", "") != "":
            assert model_config.get("controlnet_images", "") != ""
            assert model_config.get("controlnet_config", "") != ""
            
            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))

            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.cuda()

            image_paths = model_config.controlnet_images
            if isinstance(image_paths, str): image_paths = [image_paths]

            print(f"controlnet image paths:")
            for path in image_paths: print(path)
            assert len(image_paths) <= model_config.L

            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (model_config.H, model_config.W), (1.0, 1.0), 
                    ratio=(model_config.W/model_config.H, model_config.W/model_config.H)
                ),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else: image_norm = lambda x: x
                
            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]

            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            for i, image in enumerate(controlnet_images):
                Image.fromarray((255. * (image.numpy().transpose(1,2,0))).astype(np.uint8)).save(f"{savedir}/control_images/{i}.png")

            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()


        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

        pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path         = model_config.get("motion_module", ""),
            motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
            # domain adapter
            adapter_lora_path          = model_config.get("adapter_lora_path", ""),
            adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
            # image layers
            dreambooth_model_path      = model_config.get("dreambooth_path", ""),
            lora_model_path            = model_config.get("lora_model_path", ""),
            lora_alpha                 = model_config.get("lora_alpha", 0.8),
        ) #.to("cuda")

        if model_config.get("dreambooth_path", "") != "":
            savedir += "_dreambooth"

        bound_moudule = unet3d_forward.__get__(unet, unet.__class__)
        setattr(unet, "forward", bound_moudule)

        # motionctrl
        cmcm_checkpoint_path       = model_config.get("cmcm_checkpoint_path", "")
        omcm_checkpoint_path       = model_config.get("omcm_checkpoint_path", "")
        optical_flow_config        = model_config.get("optical_flow_config", None)
        if optical_flow_config is not None:
            use_optical_flow = True
        else:
            use_optical_flow = False
        # import pdb; pdb.set_trace()

        if cmcm_checkpoint_path != "" and os.path.exists(cmcm_checkpoint_path):
            name_part = cmcm_checkpoint_path.split('/')
            savedir = savedir + f"_cmcm"

            for _name, _module in unet.named_modules():
                if _module.__class__.__name__ == "TemporalTransformerBlock":
                    bound_moudule = Adapted_TemporalTransformerBlock_forward.__get__(_module, _module.__class__)
                    setattr(_module, "forward", bound_moudule)

                    cc_projection = nn.Linear(_module.attention_blocks[-1].to_k.in_features + 12, _module.attention_blocks[-1].to_k.in_features)
                    nn.init.eye_(list(cc_projection.parameters())[0][:_module.attention_blocks[-1].to_k.in_features, :_module.attention_blocks[-1].to_k.in_features])
                    nn.init.zeros_(list(cc_projection.parameters())[1])
                    cc_projection.requires_grad_(True)

                    _module.add_module('cc_projection', cc_projection)

            # load cmcm checkpoint
            print(f"load cmcm from {cmcm_checkpoint_path}")
            load_model = torch.load(cmcm_checkpoint_path, map_location="cpu")

            cmcm_state_dict = load_model["state_dict"] if "state_dict" in load_model else load_model
            new_state_dict = {}
            for k, v in cmcm_state_dict.items():
                if 'module.' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v
            cmcm_state_dict = new_state_dict
            
            cmcm_state_dict.pop("animatediff_config", "")
            missing, unexpected = pipeline.unet.load_state_dict(cmcm_state_dict, strict=False)
            assert len(unexpected) == 0
        
        pipeline = pipeline.to("cuda")

        if omcm_checkpoint_path != "" and os.path.exists(omcm_checkpoint_path):

            name_part = omcm_checkpoint_path.split('/')
            savedir = savedir + f"_omcm_{name_part[-3].split('_')[0]}"

            omcm = Adapter(**model_config.omcm_config.params)

            load_model = torch.load(omcm_checkpoint_path, map_location="cpu")
            # savedir = savedir + f"global_step{load_model['global_step']}_T{model_config.get('omcm_min_step', 700)}"

            omcm_state_dict = load_model['omcm_state_dict']
            new_state_dict = {}
            for k, v in omcm_state_dict.items():
                if 'module.' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v
            omcm_state_dict = new_state_dict

            m, u = omcm.load_state_dict(omcm_state_dict, strict=True)
            assert len(u) == 0

            idx = 0
            for _name, _module in unet.down_blocks.named_modules():
                if _module.__class__.__name__ == "CrossAttnDownBlock3D":
                    bound_moudule = Adapted_CrossAttnDownBlock3D_forward.__get__(_module, _module.__class__)
                    setattr(_module, "forward", bound_moudule)
                    setattr(_module, "traj_fea_idx", idx)
                    idx += 1

                elif _module.__class__.__name__ == "DownBlock3D":
                    bound_moudule = Adapted_DownBlock3D_forward.__get__(_module, _module.__class__)
                    setattr(_module, "forward", bound_moudule)
                    setattr(_module, "traj_fea_idx", idx)
                    idx += 1

            omcm = omcm.to(pipeline.device)

            if use_optical_flow:
                print(f'!!!!! dense optical flow !!!!!')
                opt_model = instantiate_from_config(optical_flow_config)
                assert os.path.exists(optical_flow_config.pretrained_model)
                print(f"Loading pretrained motion stage model from {optical_flow_config.pretrained_model}")
                opt_model.load_state_dict(torch.load(optical_flow_config.pretrained_model)['model'])
                opt_model.eval()
                for param in opt_model.parameters():
                    param.requires_grad = False
                num_reg_refine = optical_flow_config.num_reg_refine
                opt_model = opt_model.to(pipeline.device)

        os.makedirs(savedir, exist_ok=True)

        prompts      = model_config.prompt
        n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
        
        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
        

        RTs = []
        RT_names = []
        for RT_path in model_config.get("RT_paths", []):
            with open(RT_path, 'r') as f:
                RT = json.load(f)
            RT = np.array(RT)
            RT = torch.tensor(RT).float()# [t, 12]
            RT = RT[None, ...]
            if model_config.guidance_scale > 1.0:
                RT = torch.cat([torch.zeros_like(RT), RT], dim=0) 
            RTs.append(RT.to(pipeline.device))
            RT_name = RT_path.split("/")[-1].split(".")[0].replace("test_camera_", "")
            RT_names.append(RT_name)

        if RTs == []:
            # if cmcm_checkpoint_path != "":
            if cmcm_checkpoint_path is not None and os.path.exists(cmcm_checkpoint_path):
                RTs = [torch.zeros((2, model_config.L, 12)).to(pipeline.device)]
                RT_names.append("zero_motion")
            else:
                RTs = [None]
                RT_names.append("none_motion")

        vis_flows = []
        val_trajs = []
        val_trajs_name = []

        if use_optical_flow:
            width, height = model_config.W, model_config.H
            sample_n_frames = model_config.L
            traj_cnt = 0
            for vid_path in model_config.opt_paths:
                assert os.path.exists(vid_path), f"video path: {vid_path} does not exist"
                trajectoy = get_opt_from_video(opt_model, num_reg_refine, vid_path, width, height, sample_n_frames, device=pipeline.device)
                # cfg
                trajectoy = torch.cat([torch.zeros_like(trajectoy), trajectoy], dim=0)
                val_trajs.append(trajectoy)
                vis_flows.append(vis_opt_flow(trajectoy[1:]))
                val_trajs_name.append(f'img{traj_cnt}')
                traj_cnt += 1

        if len(model_config.get("traj_paths", [])) > 0:
            for traj_path in model_config.traj_paths:
                trajectoy = torch.tensor(np.load(traj_path)).permute(3, 0, 1, 2).float() # [t,h,w,c]->[c,t,h,w]
                trajectoy = trajectoy[None, ...]
                trajectoy = torch.cat([torch.zeros_like(trajectoy), trajectoy], dim=0)
                trajectoy = trajectoy.to(pipeline.device)
                val_trajs.append(trajectoy)
                vis_flows.append(vis_opt_flow(trajectoy[1:]))
                val_trajs_name.append(traj_path.split("/")[-1].split(".")[0])

        if val_trajs == []:
            val_trajs = [None]
            vis_flows = []
            val_trajs_name = ["no_traj"]

        config[model_idx].random_seed = []

        for traj_idx, traj in enumerate(val_trajs):
            if traj is not None:
                traj_features = get_traj_features(traj, omcm)
            else:
                traj_features = None
            for RT_idx, RT in enumerate(RTs):
                
                samples = []
                if vis_flows != []:
                    samples += [vis_flows[traj_idx]]
                for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
                    
                    # manually set random seed for reproduction
                    if random_seed != -1: torch.manual_seed(random_seed)
                    else: torch.seed()
                    config[model_idx].random_seed.append(torch.initial_seed())
                    
                    print(f"current seed: {torch.initial_seed()}")
                    print(f"sampling {prompt} ...")
                    sample = pipeline(
                        prompt,
                        negative_prompt     = n_prompt,
                        num_inference_steps = model_config.steps,
                        guidance_scale      = model_config.guidance_scale,
                        width               = model_config.W,
                        height              = model_config.H,
                        video_length        = model_config.L,

                        controlnet_images = controlnet_images,
                        controlnet_image_index = model_config.get("controlnet_image_indexs", [0]),
                        RT = RT,
                        traj_features = traj_features,
                        omcm_min_step = model_config.get("omcm_min_step", 700),
                    ).videos
                    samples.append(sample)

                    prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                    save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
                    print(f"save to {savedir}/sample/{prompt}.gif")
                    
                    sample_idx += 1

                samples = torch.concat(samples)
                save_videos_grid(samples, f"{savedir}/sample-{RT_names[RT_idx]}-{val_trajs_name[traj_idx]}.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference-config",      type=str, default="configs/inference/inference-v3.yaml")    
    parser.add_argument("--config",                type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    parser.add_argument("--without-xformers", action="store_true")

    args = parser.parse_args()
    main(args)
