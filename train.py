import argparse
import datetime
import inspect
import json
import logging
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import diffusers
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import transformers
import wandb
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from motionctrl.utils.util import instantiate_from_config
from animatediff.data.dataset import WebVid10M
from motionctrl.modified_modules import (
    Adapted_TemporalTransformerBlock_forward, unet3d_forward,
    Adapted_CrossAttnDownBlock3D_forward, Adapted_DownBlock3D_forward)
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, zero_rank_print
from motionctrl.adapter import Adapter
from motionctrl.util import get_traj_features, get_batch_motion, get_opt_from_video, vis_opt_flow


def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
        
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank


def main(
    image_finetune: bool,
    
    name: str,
    use_wandb: bool,
    launcher: str,
    
    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,
    save_checkpoint_steps: int = 10000,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,

    # cmcm
    enable_cmcm: bool = False,
    cmcm_checkpoint_path: str = "",
    appearance_debias: float = 0.0,

    # omcm
    enable_omcm: bool = False,
    omcm_config: Optional[Dict] = None,
    omcm_min_step: int = 700,
    min_step_prob: float = 0.8,

    use_optical_flow: bool = False,
    optical_flow_config: Optional[Dict] = None,
):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)
    
    # Logging folder
    if is_main_process:
        name = f'{name}_lr{learning_rate}_bs{train_batch_size}_gpus{num_processes}_ccm{gradient_accumulation_steps}_debias{appearance_debias}'
        folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
        output_dir = os.path.join(output_dir, folder_name)
        if is_debug and os.path.exists(output_dir):
            os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    if not image_finetune:
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path, subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    ################### >>>> Camera Motion Control Module >>>> ###################
    trainable_modules = []

    # cmcm
    bound_moudule = unet3d_forward.__get__(unet, unet.__class__)
    setattr(unet, "forward", bound_moudule)

    
    for _name, _module in unet.named_modules():
        if _module.__class__.__name__ == "TemporalTransformerBlock":
            bound_moudule = Adapted_TemporalTransformerBlock_forward.__get__(_module, _module.__class__)
            setattr(_module, "forward", bound_moudule)

            cc_projection = nn.Linear(_module.attention_blocks[-1].to_k.in_features + 12, _module.attention_blocks[-1].to_k.in_features)
            nn.init.eye_(list(cc_projection.parameters())[0][:_module.attention_blocks[-1].to_k.in_features, :_module.attention_blocks[-1].to_k.in_features])
            nn.init.zeros_(list(cc_projection.parameters())[1])

            _module.add_module('cc_projection', cc_projection)

            if enable_cmcm:
                cc_projection.requires_grad_(True)
                trainable_modules.append(f'{_name}.cc_projection')
                trainable_modules.append(f'{_name}.attention_blocks.1')
                trainable_modules.append(f'{_name}.norms.1')

    # omcm
    if enable_omcm:
        assert omcm_config is not None
        omcm = Adapter(**omcm_config.params)
        if omcm_config.pretrained is not None and os.path.exists(omcm_config.pretrained):
            zero_rank_print(f"Loading pretrained omcm model from {omcm_config.pretrained}")
            omcm_state_dict = torch.load(omcm_config.pretrained, map_location="cpu")['omcm_state_dict']
            new_state_dict = {}
            for k, v in omcm_state_dict.items():
                if 'module.' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v
            omcm_state_dict = new_state_dict

            m, u = omcm.load_state_dict(omcm_state_dict, strict=True)
            zero_rank_print(f"omcm missing keys: {len(m)}, omcm unexpected keys: {len(u)}")

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
        
        trainable_modules.append(omcm)

    if use_optical_flow:
        zero_rank_print(f'!!!!! dense optical flow !!!!!')
        opt_model = instantiate_from_config(optical_flow_config)
        assert os.path.exists(optical_flow_config.pretrained_model)
        zero_rank_print(f"Loading pretrained motion stage model from {optical_flow_config.pretrained_model}")
        opt_model.load_state_dict(torch.load(optical_flow_config.pretrained_model)['model'])
        opt_model.eval()
        for param in opt_model.parameters():
            param.requires_grad = False
        num_reg_refine = optical_flow_config.num_reg_refine
        opt_model = opt_model.to(local_rank)
        
    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path: zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                k = k.replace('module.', '')
            new_state_dict[k] = v
        state_dict = new_state_dict

        m, u = unet.load_state_dict(state_dict, strict=False)
        # if is_main_process:
        #     print('!!!!!!')
        #     print(m)
        #     print(u)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        # assert len(u) == 0 # do not load cmcm for memory efficiency

    if cmcm_checkpoint_path is not None and os.path.exists(cmcm_checkpoint_path):
        zero_rank_print(f"from checkpoint: {cmcm_checkpoint_path}")
        cmcm_checkpoint_path = torch.load(cmcm_checkpoint_path, map_location="cpu")
        if "global_step" in cmcm_checkpoint_path: zero_rank_print(f"global_step: {cmcm_checkpoint_path['global_step']}")
        state_dict = cmcm_checkpoint_path["state_dict"] if "state_dict" in cmcm_checkpoint_path else cmcm_checkpoint_path
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                k = k.replace('module.', '')
            new_state_dict[k] = v
        state_dict = new_state_dict

        m, u = unet.load_state_dict(state_dict, strict=False)
        zero_rank_print(f"cmcm missing keys: {len(m)}, cmcm unexpected keys: {len(u)}")
        
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Set unet trainable parameters
    trainable_params = []
    unet.requires_grad_(False)
    if enable_cmcm:
        for name, param in unet.named_parameters():
            for trainable_module_name in trainable_modules:
                if trainable_module_name in name:
                    param.requires_grad = True
                    break
        trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    
    if enable_omcm:
        trainable_params += list(omcm.parameters())

    zero_rank_print(f"trainable modules: {trainable_modules}")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params)}")
        zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            print("!!!!!!!!xformers is enabled!!!!!!!!!")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)

    # Get the training dataset
    train_dataset = instantiate_from_config(train_data)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    if is_main_process:
        # Validation pipeline
        if not image_finetune:
            validation_pipeline = AnimationPipeline(
                unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
            ).to("cuda")
        else:
            validation_pipeline = StableDiffusionPipeline.from_pretrained(
                pretrained_model_path,
                unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
            )
        validation_pipeline.enable_vae_slicing()

        val_len = len(validation_data.prompts)
        enable_guidance = validation_data.get("guidance_scale", 8.) > 1.0

        RT_paths = validation_data.get("RT_paths", [])
        if len(RT_paths) == 0:
            if cmcm_checkpoint_path != "":
                val_RTs = [torch.zeros((2, train_data.sample_n_frames, 12))] * val_len
                val_RTs = [RT.to(validation_pipeline.device) for RT in val_RTs]
            else:
                val_RTs = [None] * val_len
        else:
            assert len(RT_paths) == val_len, f'RT_paths: {len(RT_paths)} != val_len: {val_len}'
            val_RTs = []
            for RT_path in RT_paths:
                with open(RT_path, 'r') as f:
                    RT = json.load(f)
                RT = np.array(RT)
                RT = torch.tensor(RT).float()# [t, 12]
                RT = RT[None, ...]
                if enable_guidance:
                    RT = torch.cat([torch.zeros_like(RT), RT], dim=0) 
                RT = RT.to(validation_pipeline.device)
                val_RTs.append(RT)

        vis_flows = []
        traj_paths = validation_data.get("traj_paths", [])
        if len(traj_paths) == 0:
            val_trajs = [None] * val_len
            
        else:
            assert len(traj_paths) == val_len, f"traj_paths: {len(traj_paths)} != val_len: {val_len}"
            val_trajs = []
            for traj_path in validation_data.traj_paths:
                trajectoy = torch.tensor(np.load(traj_path)).permute(3, 0, 1, 2).float() # [t,h,w,c]->[c,t,h,w]
                if train_data.get("sample_size", 256) != 256:
                    trajectoy = F.interpolate(trajectoy, size=train_data.get("sample_size", 256), mode='bilinear', align_corners=False)
                trajectoy = trajectoy[None, ...]
                if enable_guidance:
                    trajectoy = torch.cat([torch.zeros_like(trajectoy), trajectoy], dim=0)
                trajectoy = trajectoy.to(validation_pipeline.device)
                val_trajs.append(trajectoy)
                vis_flows.append(vis_opt_flow(trajectoy[1:]))

        if use_optical_flow:
            val_trajs = []
            width = height = train_data.get("sample_size", 256)
            sample_n_frames = train_data.sample_n_frames 
            for vid_path in validation_data.opt_paths:
                assert os.path.exists(vid_path), f"video path: {vid_path} does not exist"
                trajectoy = get_opt_from_video(opt_model, num_reg_refine, vid_path, width, height, sample_n_frames, device=local_rank)
                if enable_guidance:
                    trajectoy = torch.cat([torch.zeros_like(trajectoy), trajectoy], dim=0)
                val_trajs.append(trajectoy)
                vis_flows.append(vis_opt_flow(trajectoy[1:]))


    # DDP warpper
    unet.to(local_rank)
    if enable_cmcm:
        unet = DDP(unet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if enable_omcm:
        omcm.to(local_rank)
        omcm = DDP(omcm, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()
        if enable_omcm:
            omcm.train()
        
        for step, batch in enumerate(train_dataloader):
            if cfg_random_null_text:
                batch['caption'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['caption']]
                
            # Data batch sanity check
            if epoch == first_epoch and step == 0 and is_main_process:
                pixel_values, texts = batch['video'].cpu(), batch['caption']
                if not image_finetune:
                    # pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w") 
                    # already done in dataset
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.gif", rescale=True)
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.png")
                    
            ### >>>> Training >>>> ###
            
            # Convert videos to latent space            
            pixel_values = batch["video"].to(local_rank)
            
            if enable_cmcm:
                RT = batch["RT"].to(local_rank)
                RT = RT[...] # [b, t, 12]
                if cfg_random_null_text:
                    for i in range(RT.shape[0]):
                        RT[i] = RT[i] if random.random() > cfg_random_null_text_ratio else torch.zeros_like(RT[i])
            else:
                RT = torch.zeros((pixel_values.shape[0], pixel_values.shape[2], 12), device=local_rank)
            
            if enable_omcm:
                if use_optical_flow:
                    trajs = get_batch_motion(opt_model, num_reg_refine, pixel_values) # [B, 2, t, H, W]
                else:
                    trajs = batch['trajs'].to(local_rank) # [b, 2, f, h, w]
                if cfg_random_null_text:
                    for i in range(trajs.shape[0]):
                        trajs[i] = trajs[i] if random.random() > cfg_random_null_text_ratio else torch.zeros_like(trajs[i])
                
                traj_features = get_traj_features(trajs, omcm)

            else:
                traj_features = None


            video_length = pixel_values.shape[2]
            with torch.no_grad():
                if not image_finetune:
                    
                    pixel_values = rearrange(pixel_values, "b c f h w -> (b f) c h w")
                    
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                    # latents = rearrange(latents, "b f c h w -> b c f h w", f=video_length)
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each video
            if omcm_min_step > 0:
                t_rand = torch.rand(bsz, device=latents.device)
                t_mask = t_rand < min_step_prob
                timesteps = torch.where(t_mask, torch.randint(omcm_min_step, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device), 
                                        torch.randint(0, omcm_min_step, (bsz,), device=latents.device))
            else:
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['caption'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]
                
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, 
                                  RT=RT, traj_features=traj_features).sample
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                if appearance_debias > 0: # appearance debias loss from MotionDirector (https://arxiv.org/abs/2310.08465)
                    anchor = []
                    for i in range(target.shape[0]):
                        randidx = random.randint(0, target.shape[2]-1)
                        anchor.append(target[i:i+1, :, randidx:randidx+1, :, :])
                    anchor = torch.cat(anchor, dim=0)
                    # repeate anchor to match target
                    anchor = anchor.repeat_interleave(target.shape[2], dim=2)
                    loss_app_debias = F.mse_loss(math.sqrt(2) * model_pred - anchor,
                                                math.sqrt(2) * target - anchor, 
                                                reduction="mean")
                    loss = loss + appearance_debias * loss_app_debias

            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1):
                save_path = os.path.join(output_dir, f"checkpoints")
                if enable_omcm:
                    omcm_state_dict = {
                        "epoch": epoch,
                        "global_step": global_step,
                        # "state_dict": unet.state_dict(),
                        "omcm_state_dict": omcm.state_dict(),
                    }
                    if step == len(train_dataloader) - 1:
                        torch.save(omcm_state_dict, os.path.join(save_path, f"omcm-epoch-{epoch+1}.ckpt"))
                    else:
                        torch.save(omcm_state_dict, os.path.join(save_path, f"omcm-checkpoint.ckpt"))
                    if global_step % save_checkpoint_steps == 0:
                        torch.save(omcm_state_dict, os.path.join(save_path, f"omcm-checkpoint-{global_step}.ckpt"))
                
                if enable_cmcm:
                    cmcm_state_dict = {}
                    for k, v in unet.state_dict().items():
                        for trainable_module_name in trainable_modules:
                            if trainable_module_name in k:
                                cmcm_state_dict[k] = v
                                break
                    state_dict = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "state_dict":cmcm_state_dict,
                    }
                    if step == len(train_dataloader) - 1:
                        torch.save(state_dict, os.path.join(save_path, f"cmcm-epoch-{epoch+1}.ckpt"))
                    else:
                        torch.save(state_dict, os.path.join(save_path, f"cmcm-checkpoint.ckpt"))

                    if global_step % save_checkpoint_steps == 0:
                        torch.save(state_dict, os.path.join(save_path, f"cmcm-checkpoint-{global_step}.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
            # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = vis_flows + []
                
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)
                
                height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                # prompts = validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) else validation_data.prompts
                prompts = validation_data.prompts

                for idx, prompt in enumerate(prompts):
                    if enable_omcm:
                        traj_features = get_traj_features(val_trajs[idx], omcm)
                    else:
                        traj_features = None
                    if not image_finetune:
                        sample = validation_pipeline(
                            prompt,
                            generator    = generator,
                            video_length = train_data.sample_n_frames,
                            height       = height,
                            width        = width,
                            RT           = val_RTs[idx],
                            traj_features = traj_features,
                            omcm_min_step = omcm_min_step,
                            **validation_data,
                        ).videos
                        save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                        samples.append(sample)
                        # import pdb; pdb.set_trace()
                        
                    else:
                        sample = validation_pipeline(
                            prompt,
                            generator           = generator,
                            height              = height,
                            width               = width,
                            num_inference_steps = validation_data.get("num_inference_steps", 25),
                            guidance_scale      = validation_data.get("guidance_scale", 8.),
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        samples.append(sample)
                        
                
                if not image_finetune:
                    samples = torch.concat(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                    save_videos_grid(samples, save_path)
                    
                else:
                    samples = torch.stack(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.png"
                    torchvision.utils.save_image(samples, save_path, nrow=4)

                logging.info(f"Saved samples to {save_path}")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",    action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
