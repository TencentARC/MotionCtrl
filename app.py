import argparse
import os
import tempfile
from functools import partial

import cv2
import gradio as gr
import imageio
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything

from gradio_utils.camera_utils import CAMERA_MOTION_MODE, process_camera
from gradio_utils.traj_utils import (OBJECT_MOTION_MODE, get_provided_traj,
                                     process_points, process_traj)
from gradio_utils.utils import vis_camera
from lvdm.models.samplers.ddim import DDIMSampler
from main.evaluation.motionctrl_inference import (DEFAULT_NEGATIVE_PROMPT,
                                                  load_model_checkpoint,
                                                  post_prompt)
from utils.utils import instantiate_from_config

os.environ['KMP_DUPLICATE_LIB_OK']='True'
SPACE_ID = os.environ.get('SPACE_ID', '')


#### Description ####
title = r"""<h1 align="center">MotionCtrl: A Unified and Flexible Motion Controller for Video Generation</h1>"""

description = r"""
<b>Official Gradio demo</b> for <a href='https://github.com/TencentARC/MotionCtrl' target='_blank'><b>MotionCtrl: A Unified and Flexible Motion Controller for Video Generation</b></a>.<br>
🔥 MotionCtrl is capable of independently and flexibly controling the camera motion and object motion of a generated video, with only a unified model.<br>
🤗 Try to control the motion of the generated videos yourself!<br>
❗❗❗ Please note that current version of **MotionCtrl** is deployed on **LVDM/VideoCrafter**. The versions that depolyed on **AnimateDiff** and **SVD** will be released soon.<br>
"""
article = r"""
If MotionCtrl is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/MotionCtrl' target='_blank'>Github Repo</a>. Thanks! 
[![GitHub Stars](https://img.shields.io/github/stars/TencentARC%2FMotionCtrl
)](https://github.com/TencentARC/MotionCtrl)

---

📝 **Citation**
<br>
If our work is useful for your research, please consider citing:
```bibtex
@inproceedings{wang2023motionctrl,
  title={MotionCtrl: A Unified and Flexible Motion Controller for Video Generation},
  author={Wang, Zhouxia and Yuan, Ziyang and Wang, Xintao and Chen, Tianshui and Xia, Menghan and Luo, Ping and Shan, Yin},
  booktitle={arXiv preprint arXiv:2312.03641},
  year={2023}
}
```

📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>wzhoux@connect.hku.hk</b>.

"""
css = """
.gradio-container {width: 85% !important}
.gr-monochrome-group {border-radius: 5px !important; border: revert-layer !important; border-width: 2px !important; color: black !important;}
span.svelte-s1r2yt {font-size: 17px !important; font-weight: bold !important; color: #d30f2f !important;}
button {border-radius: 8px !important;}
.add_button {background-color: #4CAF50 !important;}
.remove_button {background-color: #f44336 !important;}
.clear_button {background-color: gray !important;}
.mask_button_group {gap: 10px !important;}
.video {height: 300px !important;}
.image {height: 300px !important;}
.video .wrap.svelte-lcpz3o {display: flex !important; align-items: center !important; justify-content: center !important;}
.video .wrap.svelte-lcpz3o > :first-child {height: 100% !important;}
.margin_center {width: 50% !important; margin: auto !important;}
.jc_center {justify-content: center !important;}
"""


T_base = [
            [1.,0.,0.],             ## W2C  left
            [-1.,0.,0.],            ## W2C  right
            [0., 1., 0.],           ## W2C  up     
            [0.,-1.,0.],            ## W2C  down
            [0.,0.,1.],             ## W2C  zoom out
            [0.,0.,-1.],            ## W2C  zoom in
        ]   
radius = 1
n = 16
# step = 
look_at = np.array([0, 0, 0.8]).reshape(3,1)
# look_at = np.array([0, 0, 0.2]).reshape(3,1)

T_list = []
base_R = np.array([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])
res = [] 
res_forsave = []
T_range = 1.8



for i in range(0, 16):
    # theta = (1)*np.pi*i/n

    R = base_R[:,:3]
    T = np.array([0.,0.,1.]).reshape(3,1) * (i/n)*2
    RT = np.concatenate([R,T], axis=1)
    res.append(RT)
    
fig = vis_camera(res)
    
# MODE = ["camera motion control", "object motion control", "camera + object motion control"]
MODE = ["control camera poses", "control object trajectory", "control both camera and object motion"]
BASE_MODEL = ['LVDM/VideoCrafter', 'AnimateDiff', 'SVD']


traj_list = [] 
camera_dict = {
                "motion":[],
                "mode": "Customized Mode 1: First A then B",  # "First A then B", "Both A and B", "Custom"
                "speed": 1.0,
                "complex": None
                }   

def fn_vis_camera(info_mode):
    global camera_dict
    RT = process_camera(camera_dict) # [t, 3, 4]
    if camera_dict['complex'] is not None:
        # rescale T to [-2,2]
        for i in range(3):
            min_T = np.min(RT[:,i,-1])
            max_T = np.max(RT[:,i,-1])
            if min_T < -2 or max_T > 2:
                RT[:,i,-1] = RT[:,i,-1] - min_T
                RT[:,i,-1] = RT[:,i,-1] / (np.max(RT[:,:,-1]) + 1e-6)
                RT[:,i,-1] = RT[:,i,-1] * 4
                RT[:,i,-1] = RT[:,i,-1] - 2

    fig = vis_camera(RT)

    if info_mode == MODE[0]:
        vis_step3_prompt_generate = True
        vis_prompt = True
        vis_num_samples = True
        vis_seed = True
        vis_start = True
        vis_gen_video = True

        vis_object_mode = False
        vis_object_info = False

    else:
        vis_step3_prompt_generate = False
        vis_prompt = False
        vis_num_samples = False
        vis_seed = False
        vis_start = False
        vis_gen_video = False

        vis_object_mode = True
        vis_object_info = True

    return fig, \
            gr.update(visible=vis_object_mode), \
            gr.update(visible=vis_object_info), \
            gr.update(visible=vis_step3_prompt_generate), \
            gr.update(visible=vis_prompt), \
            gr.update(visible=vis_num_samples), \
            gr.update(visible=vis_seed), \
            gr.update(visible=vis_start), \
            gr.update(visible=vis_gen_video, value=None)

def fn_vis_traj():
    global traj_list
    xy_range = 1024
    points = process_points(traj_list)
    imgs = []
    for idx in range(16):
        bg_img = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
        for i in range(15):
            p = points[i]
            p1 = points[i+1]
            cv2.line(bg_img, p, p1, (255, 0, 0), 2)

            if i == idx:
                cv2.circle(bg_img, p, 2, (0, 255, 0), 20)

        if idx==(15):
            cv2.circle(bg_img, points[-1], 2, (0, 255, 0), 20)
        
        imgs.append(bg_img.astype(np.uint8))

    # size = (512, 512)
    fps = 10
    path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    writer = imageio.get_writer(path, format='mp4', mode='I', fps=fps)
    for img in imgs:
        writer.append_data(img)

    writer.close()

    vis_step3_prompt_generate = True
    vis_prompt = True
    vis_num_samples = True
    vis_seed = True
    vis_start = True
    vis_gen_video = True
    return path, gr.update(visible=vis_step3_prompt_generate), \
                gr.update(visible=vis_prompt), \
                gr.update(visible=vis_num_samples), \
                gr.update(visible=vis_seed), \
                gr.update(visible=vis_start), \
                gr.update(visible=vis_gen_video, value=None)

def display_camera_info(camera_dict, camera_mode=None):
    if camera_dict['complex'] is not None:
        res = f"complex : {camera_dict['complex']}. "
    else:
        res = ""
        res += f"motion : {[_ for _ in camera_dict['motion']]}. "
        res += f"speed : {camera_dict['speed']}. "
        if camera_mode == CAMERA_MOTION_MODE[2]:
            res += f"mode : {camera_dict['mode']}. "
    return res

def add_traj_point(evt: gr.SelectData, ):
    global traj_list
    traj_list.append(evt.index)
    traj_str = [f"{traj}" for traj in traj_list]
    return ", ".join(traj_str)

def add_provided_traj(traj_name):
    global traj_list
    traj_list = get_provided_traj(traj_name)
    traj_str = [f"{traj}" for traj in traj_list]
    return ", ".join(traj_str)

def add_camera_motion(camera_motion, camera_mode):  
    global camera_dict
    if camera_dict['complex'] is not None:
        camera_dict['complex'] = None
    if camera_mode == CAMERA_MOTION_MODE[2] and len(camera_dict['motion']) <2:
        camera_dict['motion'].append(camera_motion)
    else:
        camera_dict['motion']=[camera_motion]
    
    return display_camera_info(camera_dict, camera_mode)

def add_complex_camera_motion(camera_motion):
    global camera_dict
    camera_dict['complex']=camera_motion
    return display_camera_info(camera_dict)

def change_camera_mode(combine_type, camera_mode):
    global camera_dict
    camera_dict['mode'] = combine_type

    return display_camera_info(camera_dict, camera_mode)

def change_camera_speed(camera_speed):
    global camera_dict
    camera_dict['speed'] = camera_speed
    return display_camera_info(camera_dict)

def reset_camera():
    global camera_dict
    camera_dict = {
                    "motion":[],
                    "mode": "Customized Mode 1: First A then B",
                    "speed": 1.0,
                    "complex": None
                    }   
    return display_camera_info(camera_dict)


def fn_traj_droplast():
    global traj_list

    if traj_list:
        traj_list.pop()

    if traj_list:
        traj_str = [f"{traj}" for traj in traj_list]
        return ", ".join(traj_str)
    else:   
        return "Click to specify trajectory"

def fn_traj_reset():
    global traj_list
    traj_list = []
    return "Click to specify trajectory"

###########################################

model_path='./checkpoints/motionctrl.pth'
config_path='./configs/inference/config_both.yaml'

config = OmegaConf.load(config_path)
model_config = config.pop("model", OmegaConf.create())
model = instantiate_from_config(model_config)
if torch.cuda.is_available():
    model = model.cuda()

model = load_model_checkpoint(model, model_path)
model.eval()


def model_run(prompts, infer_mode, seed, n_samples):
    global traj_list
    global camera_dict

    RT = process_camera(camera_dict).reshape(-1,12)
    traj_flow = process_traj(traj_list).transpose(3,0,1,2)
    print(prompts)
    print(RT.shape)
    print(traj_flow.shape)

    noise_shape = [1, 4, 16, 32, 32]
    unconditional_guidance_scale = 7.5
    unconditional_guidance_scale_temporal = None
    # n_samples = 1
    ddim_steps= 50
    ddim_eta=1.0
    cond_T=800

    if n_samples < 1:
        n_samples = 1
    if n_samples > 4:
        n_samples = 4

    seed_everything(seed)

    if infer_mode == MODE[0]:
        camera_poses = RT
        camera_poses = torch.tensor(camera_poses).float()
        camera_poses = camera_poses.unsqueeze(0)
        trajs = None
        if torch.cuda.is_available():
            camera_poses = camera_poses.cuda()
    elif infer_mode == MODE[1]:
        trajs = traj_flow
        trajs = torch.tensor(trajs).float()
        trajs = trajs.unsqueeze(0)
        camera_poses = None
        if torch.cuda.is_available():
            trajs = trajs.cuda()
    else:
        camera_poses = RT
        trajs = traj_flow
        camera_poses = torch.tensor(camera_poses).float()
        trajs = torch.tensor(trajs).float()
        camera_poses = camera_poses.unsqueeze(0)
        trajs = trajs.unsqueeze(0)
        if torch.cuda.is_available():
            camera_poses = camera_poses.cuda()
            trajs = trajs.cuda()


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
                                            cond_T=cond_T
                                            )        
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    batch_variants = batch_variants[0]
    
    # file_path = save_results(batch_variants, "MotionCtrl", "gradio_temp", fps=10)
    file_path = save_results(batch_variants, fps=10)
    print(file_path)

    return gr.update(value=file_path, width=256*n_samples, height=256)

    # return file_path

def save_results(video, fps=10):
    
    # b,c,t,h,w
    video = video.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
    frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n)) for framesheet in video] #[3, 1*h, n*w]
    grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [t, h, w*n, 3]
    
    path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

    writer = imageio.get_writer(path, format='mp4', mode='I', fps=fps)
    for i in range(grid.shape[0]):
        img = grid[i].numpy()
        writer.append_data(img)

    writer.close()

    return path

def visualized_step2(infer_mode):

    # reset
    reset_camera()
    fn_traj_reset()

    # camera motion control
    vis_basic_camera_motion = False
    vis_basic_camera_motion_des = False
    vis_custom_camera_motion = False
    vis_custom_run_status = False
    vis_complex_camera_motion = False
    vis_complex_camera_motion_des = False
    vis_U = False
    vis_D = False
    vis_L = False
    vis_R = False
    vis_I = False
    vis_O = False
    vis_ACW = False
    vis_CW = False
    vis_combine1 = False
    vis_combine2 = False
    vis_speed = False

    vis_Pose_1, vis_Pose_2, vis_Pose_3, vis_Pose_4 = False, False, False, False
    vis_Pose_5, vis_Pose_6, vis_Pose_7, vis_Pose_8 = False, False, False, False

    vis_camera_args = False
    vis_camera_reset = False
    vis_camera_vis = False
    vis_vis_camera = False

    # object motion control
    vis_provided_traj = False
    vis_provided_traj_des = False
    vis_draw_yourself = False
    vis_draw_run_status = False

    vis_traj_1, vis_traj_2, vis_traj_3, vis_traj_4 = False, False, False, False
    vis_traj_5, vis_traj_6, vis_traj_7, vis_traj_8 = False, False, False, False

    traj_args = False
    traj_droplast, traj_reset = False, False
    traj_vis = False
    traj_input, vis_traj = False, False


    # generate video
    vis_step3_prompt_generate = False
    vis_prompt = False
    vis_num_samples = False
    vis_seed = False
    vis_start = False
    vis_gen_video = False

    if infer_mode == MODE[0]:
        vis_step2_camera_motion = True
        vis_step2_camera_motion_des = True
        vis_camera_mode = True
        vis_camera_info = True

        vis_step2_object_motion = False
        vis_step2_object_motion_des = False
        vis_traj_mode = False
        vis_traj_info = False

        step2_camera_object_motion = False
        step2_camera_object_motion_des = False

    elif infer_mode == MODE[1]:
        vis_step2_camera_motion = False
        vis_step2_camera_motion_des = False
        vis_camera_mode = False
        vis_camera_info = False

        vis_step2_object_motion = True
        vis_step2_object_motion_des = True
        vis_traj_mode = True
        vis_traj_info = True

        step2_camera_object_motion = False
        step2_camera_object_motion_des = False
    else: #infer_mode == MODE[2]:
        vis_step2_camera_motion = False
        vis_step2_camera_motion_des = False
        vis_camera_mode = False
        vis_camera_info = False
    
        vis_step2_object_motion = False
        vis_step2_object_motion_des = False
        vis_traj_mode = False
        vis_traj_info = False

        step2_camera_object_motion = True
        step2_camera_object_motion_des = True
    
        vis_basic_camera_motion = True
        vis_basic_camera_motion_des = True
        vis_U = True
        vis_D = True
        vis_L = True
        vis_R = True
        vis_I = True
        vis_O = True
        vis_ACW = True
        vis_CW = True
        vis_speed = True

        vis_camera_args = True
        vis_camera_reset = True
        vis_camera_vis = True
        vis_vis_camera = True
        
    
    return gr.update(visible=vis_step2_camera_motion), \
            gr.update(visible=vis_step2_camera_motion_des), \
            gr.update(visible=vis_camera_mode), \
            gr.update(visible=vis_camera_info), \
            gr.update(visible=vis_basic_camera_motion), \
            gr.update(visible=vis_basic_camera_motion_des), \
            gr.update(visible=vis_custom_camera_motion), \
            gr.update(visible=vis_custom_run_status), \
            gr.update(visible=vis_complex_camera_motion), \
            gr.update(visible=vis_complex_camera_motion_des), \
            gr.update(visible=vis_U), gr.update(visible=vis_D), gr.update(visible=vis_L), gr.update(visible=vis_R), \
            gr.update(visible=vis_I), gr.update(visible=vis_O), gr.update(visible=vis_ACW), gr.update(visible=vis_CW), \
            gr.update(visible=vis_combine1), gr.update(visible=vis_combine2), \
            gr.update(visible=vis_speed), \
            gr.update(visible=vis_Pose_1), gr.update(visible=vis_Pose_2), gr.update(visible=vis_Pose_3), gr.update(visible=vis_Pose_4), \
            gr.update(visible=vis_Pose_5), gr.update(visible=vis_Pose_6), gr.update(visible=vis_Pose_7), gr.update(visible=vis_Pose_8), \
            gr.update(visible=vis_camera_args, value=None), \
            gr.update(visible=vis_camera_reset), gr.update(visible=vis_camera_vis), \
            gr.update(visible=vis_vis_camera, value=None), \
            gr.update(visible=vis_step2_object_motion), \
            gr.update(visible=vis_step2_object_motion_des), \
            gr.update(visible=vis_traj_mode), \
            gr.update(visible=vis_traj_info), \
            gr.update(visible=vis_provided_traj), \
            gr.update(visible=vis_provided_traj_des), \
            gr.update(visible=vis_draw_yourself), \
            gr.update(visible=vis_draw_run_status), \
            gr.update(visible=vis_traj_1), gr.update(visible=vis_traj_2), gr.update(visible=vis_traj_3), gr.update(visible=vis_traj_4), \
            gr.update(visible=vis_traj_5), gr.update(visible=vis_traj_6), gr.update(visible=vis_traj_7), gr.update(visible=vis_traj_8), \
            gr.update(visible=traj_args), \
            gr.update(visible=traj_droplast), gr.update(visible=traj_reset), \
            gr.update(visible=traj_vis), \
            gr.update(visible=traj_input), gr.update(visible=vis_traj, value=None), \
            gr.update(visible=step2_camera_object_motion), \
            gr.update(visible=step2_camera_object_motion_des), \
            gr.update(visible=vis_step3_prompt_generate), \
            gr.update(visible=vis_prompt), \
            gr.update(visible=vis_num_samples), \
            gr.update(visible=vis_seed), \
            gr.update(visible=vis_start), \
            gr.update(visible=vis_gen_video)

def visualized_camera_poses(step2_camera_motion):
    reset_camera()

    # generate video
    vis_step3_prompt_generate = False
    vis_prompt = False
    vis_num_samples = False
    vis_seed = False
    vis_start = False
    vis_gen_video = False

    if step2_camera_motion == CAMERA_MOTION_MODE[0]:
        vis_basic_camera_motion = True
        vis_basic_camera_motion_des = True
        vis_custom_camera_motion = False
        vis_custom_run_status = False
        vis_complex_camera_motion = False
        vis_complex_camera_motion_des = False
        vis_U = True
        vis_D = True
        vis_L = True
        vis_R = True
        vis_I = True
        vis_O = True
        vis_ACW = True
        vis_CW = True
        vis_combine1 = False
        vis_combine2 = False
        vis_speed = True

        vis_Pose_1, vis_Pose_2, vis_Pose_3, vis_Pose_4 = False, False, False, False
        vis_Pose_5, vis_Pose_6, vis_Pose_7, vis_Pose_8 = False, False, False, False

    elif step2_camera_motion == CAMERA_MOTION_MODE[1]:
        vis_basic_camera_motion = False
        vis_basic_camera_motion_des = False
        vis_custom_camera_motion = False
        vis_custom_run_status = False
        vis_complex_camera_motion = True
        vis_complex_camera_motion_des = True
        vis_U = False
        vis_D = False
        vis_L = False
        vis_R = False
        vis_I = False
        vis_O = False
        vis_ACW = False
        vis_CW = False
        vis_combine1 = False
        vis_combine2 = False
        vis_speed = False

        vis_Pose_1, vis_Pose_2, vis_Pose_3, vis_Pose_4 = True, True, True, True
        vis_Pose_5, vis_Pose_6, vis_Pose_7, vis_Pose_8 = True, True, True, True

    else: # step2_camera_motion = CAMERA_MOTION_MODE[2]:
        vis_basic_camera_motion = False
        vis_basic_camera_motion_des = False
        vis_custom_camera_motion = True
        vis_custom_run_status = True
        vis_complex_camera_motion = False
        vis_complex_camera_motion_des = False
        vis_U = True
        vis_D = True
        vis_L = True
        vis_R = True
        vis_I = True
        vis_O = True
        vis_ACW = True
        vis_CW = True
        vis_combine1 = True
        vis_combine2 = True
        vis_speed = True

        vis_Pose_1, vis_Pose_2, vis_Pose_3, vis_Pose_4 = False, False, False, False
        vis_Pose_5, vis_Pose_6, vis_Pose_7, vis_Pose_8 = False, False, False, False

    vis_camera_args = True
    vis_camera_reset = True
    vis_camera_vis = True
    vis_vis_camera = True

    return gr.update(visible=vis_basic_camera_motion), \
            gr.update(visible=vis_basic_camera_motion_des), \
            gr.update(visible=vis_custom_camera_motion), \
            gr.update(visible=vis_custom_run_status), \
            gr.update(visible=vis_complex_camera_motion), \
            gr.update(visible=vis_complex_camera_motion_des), \
            gr.update(visible=vis_U), gr.update(visible=vis_D), gr.update(visible=vis_L), gr.update(visible=vis_R), \
            gr.update(visible=vis_I), gr.update(visible=vis_O), gr.update(visible=vis_ACW), gr.update(visible=vis_CW), \
            gr.update(visible=vis_combine1), gr.update(visible=vis_combine2), \
            gr.update(visible=vis_speed), \
            gr.update(visible=vis_Pose_1), gr.update(visible=vis_Pose_2), gr.update(visible=vis_Pose_3), gr.update(visible=vis_Pose_4), \
            gr.update(visible=vis_Pose_5), gr.update(visible=vis_Pose_6), gr.update(visible=vis_Pose_7), gr.update(visible=vis_Pose_8), \
            gr.update(visible=vis_camera_args, value=None), \
            gr.update(visible=vis_camera_reset), gr.update(visible=vis_camera_vis), \
            gr.update(visible=vis_vis_camera, value=None), \
            gr.update(visible=vis_step3_prompt_generate), \
            gr.update(visible=vis_prompt), \
            gr.update(visible=vis_num_samples), \
            gr.update(visible=vis_seed), \
            gr.update(visible=vis_start), \
            gr.update(visible=vis_gen_video)

def visualized_traj_poses(step2_object_motion):
    
    fn_traj_reset()

    # generate video
    vis_step3_prompt_generate = False
    vis_prompt = False
    vis_num_samples = False
    vis_seed = False
    vis_start = False
    vis_gen_video = False

    if step2_object_motion == "Provided Trajectory":
        vis_provided_traj = True
        vis_provided_traj_des = True
        vis_draw_yourself = False
        vis_draw_run_status = False

        vis_traj_1, vis_traj_2, vis_traj_3, vis_traj_4 = True, True, True, True
        vis_traj_5, vis_traj_6, vis_traj_7, vis_traj_8 = True, True, True, True

        traj_args = True
        traj_droplast, traj_reset = False, True
        traj_vis = True
        traj_input, vis_traj = False, True


    elif step2_object_motion == "Custom Trajectory":
        vis_provided_traj = False
        vis_provided_traj_des = False
        vis_draw_yourself = True
        vis_draw_run_status = True

        vis_traj_1, vis_traj_2, vis_traj_3, vis_traj_4 = False, False, False, False
        vis_traj_5, vis_traj_6, vis_traj_7, vis_traj_8 = False, False, False, False

        traj_args = True
        traj_droplast, traj_reset = True, True
        traj_vis = True
        traj_input, vis_traj = True, True

    return gr.update(visible=vis_provided_traj), \
            gr.update(visible=vis_provided_traj_des), \
            gr.update(visible=vis_draw_yourself), \
            gr.update(visible=vis_draw_run_status), \
            gr.update(visible=vis_traj_1), gr.update(visible=vis_traj_2), gr.update(visible=vis_traj_3), gr.update(visible=vis_traj_4), \
            gr.update(visible=vis_traj_5), gr.update(visible=vis_traj_6), gr.update(visible=vis_traj_7), gr.update(visible=vis_traj_8), \
            gr.update(visible=traj_args), \
            gr.update(visible=traj_droplast), gr.update(visible=traj_reset), \
            gr.update(visible=traj_vis), \
            gr.update(visible=traj_input), gr.update(visible=vis_traj, value=None), \
            gr.update(visible=vis_step3_prompt_generate), \
            gr.update(visible=vis_prompt), \
            gr.update(visible=vis_num_samples), \
            gr.update(visible=vis_seed), \
            gr.update(visible=vis_start), \
            gr.update(visible=vis_gen_video)

def main(args):
    demo = gr.Blocks()
    with demo:

        gr.Markdown(title)
        gr.Markdown(description)

        # state = gr.State({
        #     "mode": "camera_only",
        #     "camera_input": [],
        #     "traj_input": [],
        # })

        with gr.Column():
            '''
            # step 0: select based model.
            gr.Markdown("## Step0: Selecting the model", show_label=False)
            gr.Markdown( f'- {BASE_MODEL[0]}: **MotionCtrl** deployed on {BASE_MODEL[0]}', show_label=False)
            gr.Markdown( f'- {BASE_MODEL[1]}: **MotionCtrl** deployed on {BASE_MODEL[1]}', show_label=False)
            gr.Markdown( f'- {BASE_MODEL[2]}: **MotionCtrl** deployed on {BASE_MODEL[2]}', show_label=False)
            gr.Markdown( f'- **Only the model that deployed on {BASE_MODEL[0]} is avalible now. MotionCtrl models deployed on {BASE_MODEL[1]} and {BASE_MODEL[2]} are coming soon.**', show_label=False)
            gr.Radio(choices=BASE_MODEL, value=BASE_MODEL[0], label="Based Model", interactive=False)
            '''

            # step 1: select motion control mode
            gr.Markdown("## Step 1/3: Selecting the motion control mode", show_label=False)
            gr.Markdown( f'- {MODE[0]}: Control the camera motion only', show_label=False)
            gr.Markdown( f'- {MODE[1]}: Control the object motion only', show_label=False)
            gr.Markdown( f'- {MODE[2]}: Control both the camera and object motion', show_label=False)
            gr.Markdown( f'- Click `Proceed` to go into next step', show_label=False)
            infer_mode = gr.Radio(choices=MODE, value=MODE[0], label="Motion Control Mode", interactive=True)
            mode_info = gr.Button(value="Proceed")

            # step2 - camera + object motion control
            step2_camera_object_motion  = gr.Markdown("---\n## Step 2/3: Select the camera poses and trajectory", show_label=False, visible=False)
            step2_camera_object_motion_des = gr.Markdown(f"\n 1. Select a basic camera pose. \
                                                            \n 2. Select a provided trajectory or draw the trajectory yourself.",
                                                        show_label=False, visible=False)
        
            # step2 - camera motion control
            step2_camera_motion = gr.Markdown("---\n## Step 2/3: Select the camera poses", show_label=False, visible=False)
            step2_camera_motion_des = gr.Markdown(f"\n - {CAMERA_MOTION_MODE[0]}: Including 8 basic camera poses, such as pan up, pan down, zoom in, and zoom out. \
                                                    \n - {CAMERA_MOTION_MODE[1]}: Complex camera poses extracted from the real videos. \
                                                    \n - {CAMERA_MOTION_MODE[2]}: You can customize complex camera poses yourself by combining or fusing two of the eight basic camera poses. \
                                                    \n - Click `Proceed` to go into next step", 
                                                  show_label=False, visible=False)
            camera_mode = gr.Radio(choices=CAMERA_MOTION_MODE, value=CAMERA_MOTION_MODE[0], label="Camera Motion Control Mode", interactive=True, visible=False)
            camera_info = gr.Button(value="Proceed", visible=False)

            with gr.Row():
                with gr.Column():
                    # step2.1 - camera motion control - basic
                    basic_camera_motion = gr.Markdown("---\n### Basic Camera Poses", show_label=False, visible=False)
                    basic_camera_motion_des = gr.Markdown(f"\n 1. Click one of the basic camera poses, such as `Pan Up`; \
                                                            \n 2. Slide the `Motion speed` to get a speed value. The large the value, the fast the camera motion; \
                                                            \n 3. Click `Visualize Camera and Proceed` to visualize the camera poses and go proceed; \
                                                            \n 4. Click `Reset Camera` to reset the camera poses (If needed). ",
                                                        show_label=False, visible=False)
                    
                    
                    # step2.2 - camera motion control - provided complex
                    complex_camera_motion = gr.Markdown("---\n### Provided Complex Camera Poses", show_label=False, visible=False)
                    complex_camera_motion_des = gr.Markdown(f"\n 1. Click one of the complex camera poses, such as `Pose_1`; \
                                                            \n 2. Click `Visualize Camera and Proceed` to visualize the camera poses and go proceed; \
                                                            \n 3. Click `Reset Camera` to reset the camera poses (If needed). ",
                                                        show_label=False, visible=False)

                    # step2.3 - camera motion control - custom
                    custom_camera_motion = gr.Markdown(f"---\n### {CAMERA_MOTION_MODE[2]}", show_label=False, visible=False)
                    custom_run_status = gr.Markdown(f"\n 1. Click two of the basic camera poses, such as `Pan Up` and `Pan Left`; \
                                                    \n 2. Click `Customized Mode 1: First A then B` or `Customized Mode 1: First A then B` \
                                                    \n - `Customized Mode 1: First A then B`: The camera first `Pan Up` and then `Pan Left`; \
                                                    \n - `Customized Mode 2: Both A and B`: The camera move towards the upper left corner; \
                                                    \n 3. Slide the `Motion speed` to get a speed value. The large the value, the fast the camera motion; \
                                                    \n 4. Click `Visualize Camera and Proceed` to visualize the camera poses and go proceed; \
                                                    \n 5. Click `Reset Camera` to reset the camera poses (If needed). ",
                                                        show_label=False, visible=False)

                    gr.HighlightedText(value=[("",""), ("1. Select two of the basic camera poses; 2. Select Customized Mode 1 OR Customized Mode 2. 3. Visualized Camera to show the customized camera poses", "Normal")],
                                                        color_map={"Normal": "green", "Error": "red", "Clear clicks": "gray", "Add mask": "green", "Remove mask": "red"}, visible=False)
                    
                    with gr.Row():
                        U = gr.Button(value="Pan Up", visible=False)
                        D = gr.Button(value="Pan Down", visible=False)
                        L = gr.Button(value="Pan Left", visible=False)
                        R = gr.Button(value="Pan Right", visible=False)
                    with gr.Row():
                        I = gr.Button(value="Zoom In", visible=False)
                        O = gr.Button(value="Zoom Out", visible=False)
                        ACW = gr.Button(value="ACW", visible=False)
                        CW = gr.Button(value="CW", visible=False)
                    
                    with gr.Row():
                        combine1 = gr.Button(value="Customized Mode 1: First A then B", visible=False)
                        combine2 = gr.Button(value="Customized Mode 2: Both A and B", visible=False)

                    with gr.Row():    
                        speed = gr.Slider(minimum=0, maximum=2, step=0.2, label="Motion Speed", value=1.0, visible=False)

                    with gr.Row():
                        Pose_1 = gr.Button(value="Pose_1", visible=False)
                        Pose_2 = gr.Button(value="Pose_2", visible=False)
                        Pose_3 = gr.Button(value="Pose_3", visible=False)
                        Pose_4 = gr.Button(value="Pose_4", visible=False)
                    with gr.Row():
                        Pose_5 = gr.Button(value="Pose_5", visible=False)
                        Pose_6 = gr.Button(value="Pose_6", visible=False)
                        Pose_7 = gr.Button(value="Pose_7", visible=False)
                        Pose_8 = gr.Button(value="Pose_8", visible=False)
                
                    with gr.Row():
                        camera_args = gr.Textbox(value="Camera Type", label="Camera Type", visible=False)
                    with gr.Row():
                        camera_vis= gr.Button(value="Visualize Camera and Proceed", visible=False)
                        camera_reset = gr.Button(value="Reset Camera", visible=False)
                with gr.Column():
                    vis_camera = gr.Plot(fig, label='Camera Poses', visible=False)

            # step2 - object motion control
            step2_object_motion = gr.Markdown("---\n## Step 2/3: Select a Provided Trajectory of Draw Yourself", show_label=False, visible=False)
            step2_object_motion_des = gr.Markdown(f"\n - {OBJECT_MOTION_MODE[0]}: We provide some example trajectories. You can select one of them directly. \
                                                    \n - {OBJECT_MOTION_MODE[1]}: Draw the trajectory yourself. \
                                                    \n - Click `Proceed` to go into next step", 
                                                  show_label=False, visible=False)

            object_mode = gr.Radio(choices=OBJECT_MOTION_MODE, value=OBJECT_MOTION_MODE[0], label="Motion Control Mode", interactive=True, visible=False)
            object_info = gr.Button(value="Proceed", visible=False)

            with gr.Row():
                with gr.Column():
                    # step2.1 - object motion control - provided
                    provided_traj = gr.Markdown("---\n### Provided Trajectory", show_label=False, visible=False)
                    provided_traj_des = gr.Markdown(f"\n 1. Click one of the provided trajectories, such as `horizon_1`; \
                                                      \n 2. Click `Visualize Trajectory and Proceed` to visualize the trajectory and go proceed; \
                                                        \n 3. Click `Reset Trajectory` to reset the trajectory (If needed). ",
                                                        show_label=False, visible=False)

                    # step2.2 - object motion control - draw yourself
                    draw_traj = gr.Markdown("---\n### Draw Yourself", show_label=False, visible=False)
                    draw_run_status = gr.Markdown(f"\n 1. Click the `Canvas` in the right to draw the trajectory. **Note that You have to click the canva many times. For time saving, \
                                                  the click point will not appear in the canvas but its coordinates will be written in `Points of Trajectory`**; \
                                                  \n 2. Click `Visualize Trajectory and Proceed` to visualize the trajectory and go proceed; \
                                                  \n 3. Click `Reset Trajectory` to reset the trajectory (If needed). ",
                                                    show_label=False, visible=False)
                    
                    with gr.Row():
                        traj_1 = gr.Button(value="horizon_1", visible=False)
                        traj_2 = gr.Button(value="swaying_1", visible=False)
                        traj_3 = gr.Button(value="swaying_2", visible=False)
                        traj_4 = gr.Button(value="swaying_3", visible=False)
                    with gr.Row():
                        traj_5 = gr.Button(value="curve_1", visible=False)
                        traj_6 = gr.Button(value="curve_2", visible=False)
                        traj_7 = gr.Button(value="curve_3", visible=False)
                        traj_8 = gr.Button(value="curve_4", visible=False)

                    traj_args = gr.Textbox(value="", label="Points of Trajectory", visible=False)
                    with gr.Row():
                        traj_vis = gr.Button(value="Visualize Trajectory and Proceed", visible=False)
                        traj_reset = gr.Button(value="Reset Trajectory", visible=False)
                        traj_droplast = gr.Button(value="Drop Last Point", visible=False)
                
                with gr.Column():
                    traj_input = gr.Image("assets/traj_layout.png", tool='sketch', source="canvas", 
                                    width=256, height=256,
                                    label="Canvas for Drawing", visible=False)
                    
                    vis_traj = gr.Video(value=None, label="Trajectory", visible=False, width=256, height=256)



            # step3 - Add prompt and Generate videos
            with gr.Row():
                with gr.Column():
                    step3_prompt_generate = gr.Markdown("---\n## Step 3/3: Add prompt and Generate videos", show_label=False, visible=False)
                    prompt = gr.Textbox(value="a dog sitting on grass", label="Prompt", interactive=True, visible=False)
                    n_samples = gr.Number(value=3, precision=0, interactive=True, label="n_samples", visible=False)
                    seed = gr.Number(value=1234, precision=0, interactive=True, label="Seed", visible=False)
                    start = gr.Button(value="Start generation !", visible=False)
                with gr.Column():
                    gen_video = gr.Video(value=None, label="Generate Video", visible=False)

        mode_info.click(
            fn=visualized_step2,
            inputs=[infer_mode],
            outputs=[step2_camera_motion, 
                     step2_camera_motion_des,
                     camera_mode, 
                     camera_info,

                     basic_camera_motion,
                     basic_camera_motion_des,
                     custom_camera_motion,
                     custom_run_status,
                     complex_camera_motion,
                     complex_camera_motion_des,
                     U, D, L, R, 
                     I, O, ACW, CW, 
                     combine1, combine2,
                     speed, 
                     Pose_1, Pose_2, Pose_3, Pose_4, 
                     Pose_5, Pose_6, Pose_7, Pose_8,
                     camera_args, 
                     camera_reset, camera_vis,
                     vis_camera,

                     step2_object_motion,
                     step2_object_motion_des,
                     object_mode,
                     object_info,

                    provided_traj,
                    provided_traj_des,
                    draw_traj,
                    draw_run_status,
                    traj_1, traj_2, traj_3, traj_4,
                    traj_5, traj_6, traj_7, traj_8,
                    traj_args,
                    traj_droplast, traj_reset,
                    traj_vis,
                    traj_input, vis_traj,

                    step2_camera_object_motion,
                    step2_camera_object_motion_des,

                     step3_prompt_generate, prompt, n_samples, seed, start, gen_video,
                     
                     ],
        )

        camera_info.click(
            fn=visualized_camera_poses,
            inputs=[camera_mode],
            outputs=[basic_camera_motion,
                     basic_camera_motion_des,
                     custom_camera_motion,
                     custom_run_status,
                     complex_camera_motion,
                     complex_camera_motion_des,
                     U, D, L, R, 
                     I, O, ACW, CW, 
                     combine1, combine2,
                     speed, 
                     Pose_1, Pose_2, Pose_3, Pose_4, 
                     Pose_5, Pose_6, Pose_7, Pose_8,
                     camera_args, 
                     camera_reset, camera_vis,
                     vis_camera,
                     step3_prompt_generate, prompt, n_samples, seed, start, gen_video],
        )

        object_info.click(
            fn=visualized_traj_poses,
            inputs=[object_mode],
            outputs=[provided_traj,
                     provided_traj_des,
                        draw_traj,
                        draw_run_status,
                        traj_1, traj_2, traj_3, traj_4,
                        traj_5, traj_6, traj_7, traj_8,
                        traj_args,
                        traj_droplast, traj_reset,
                        traj_vis,
                        traj_input, vis_traj,
                        step3_prompt_generate, prompt, n_samples, seed, start, gen_video,],
        )


        U.click(fn=add_camera_motion, inputs=[U, camera_mode], outputs=camera_args)
        D.click(fn=add_camera_motion, inputs=[D, camera_mode], outputs=camera_args)
        L.click(fn=add_camera_motion, inputs=[L, camera_mode], outputs=camera_args)
        R.click(fn=add_camera_motion, inputs=[R, camera_mode], outputs=camera_args)
        I.click(fn=add_camera_motion, inputs=[I, camera_mode], outputs=camera_args)
        O.click(fn=add_camera_motion, inputs=[O, camera_mode], outputs=camera_args)
        ACW.click(fn=add_camera_motion, inputs=[ACW, camera_mode], outputs=camera_args)
        CW.click(fn=add_camera_motion, inputs=[CW, camera_mode], outputs=camera_args)
        speed.change(fn=change_camera_speed, inputs=speed, outputs=camera_args)
        camera_reset.click(fn=reset_camera, inputs=None, outputs=[camera_args])

        combine1.click(fn=change_camera_mode, inputs=[combine1, camera_mode], outputs=camera_args)
        combine2.click(fn=change_camera_mode, inputs=[combine2, camera_mode], outputs=camera_args)

        camera_vis.click(fn=fn_vis_camera, inputs=[infer_mode], outputs=[vis_camera, object_mode, object_info, step3_prompt_generate, prompt, n_samples, seed, start, gen_video])

        Pose_1.click(fn=add_complex_camera_motion, inputs=Pose_1, outputs=camera_args)
        Pose_2.click(fn=add_complex_camera_motion, inputs=Pose_2, outputs=camera_args)
        Pose_3.click(fn=add_complex_camera_motion, inputs=Pose_3, outputs=camera_args)
        Pose_4.click(fn=add_complex_camera_motion, inputs=Pose_4, outputs=camera_args)
        Pose_5.click(fn=add_complex_camera_motion, inputs=Pose_5, outputs=camera_args)
        Pose_6.click(fn=add_complex_camera_motion, inputs=Pose_6, outputs=camera_args)
        Pose_7.click(fn=add_complex_camera_motion, inputs=Pose_7, outputs=camera_args)
        Pose_8.click(fn=add_complex_camera_motion, inputs=Pose_8, outputs=camera_args)

        traj_1.click(fn=add_provided_traj, inputs=traj_1, outputs=traj_args)
        traj_2.click(fn=add_provided_traj, inputs=traj_2, outputs=traj_args)
        traj_3.click(fn=add_provided_traj, inputs=traj_3, outputs=traj_args)
        traj_4.click(fn=add_provided_traj, inputs=traj_4, outputs=traj_args)
        traj_5.click(fn=add_provided_traj, inputs=traj_5, outputs=traj_args)
        traj_6.click(fn=add_provided_traj, inputs=traj_6, outputs=traj_args)
        traj_7.click(fn=add_provided_traj, inputs=traj_7, outputs=traj_args)
        traj_8.click(fn=add_provided_traj, inputs=traj_8, outputs=traj_args)

        traj_vis.click(fn=fn_vis_traj, inputs=None, outputs=[vis_traj, step3_prompt_generate, prompt, n_samples, seed, start, gen_video])
        traj_input.select(fn=add_traj_point, inputs=None, outputs=traj_args)
        traj_droplast.click(fn=fn_traj_droplast, inputs=None, outputs=traj_args)
        traj_reset.click(fn=fn_traj_reset, inputs=None, outputs=traj_args)


        start.click(fn=model_run, inputs=[prompt, infer_mode, seed, n_samples], outputs=gen_video)

        gr.Markdown(article)

    # demo.launch(server_name='0.0.0.0', share=False, server_port=args.port)
    # demo.queue(concurrency_count=1, max_size=10)
    # demo.launch()
    demo.queue(max_size=10).launch(**args)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--port", type=int, default=12345)

    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    main(launch_kwargs)
