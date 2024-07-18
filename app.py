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
from gradio_utils.traj_utils import (OBJECT_MOTION_MODE, process_traj)
from gradio_utils.utils import vis_camera
from lvdm.models.samplers.ddim import DDIMSampler
from main.evaluation.motionctrl_inference import (DEFAULT_NEGATIVE_PROMPT,
                                                  load_model_checkpoint,
                                                  post_prompt)
from utils.utils import instantiate_from_config

from gradio_utils.page_control import (MODE, BASE_MODEL, 
                                       get_camera_dict, get_traj_list,
                                       reset_camera, 
                                       visualized_step1, visualized_step2,
                                       visualized_camera_poses, visualized_traj_poses,
                                       add_camera_motion, add_complex_camera_motion, 
                                       input_raw_camera_pose,
                                       change_camera_mode, change_camera_speed,
                                       add_traj_point, add_provided_traj, 
                                       fn_traj_droplast, fn_traj_reset,
                                       fn_vis_camera, fn_vis_traj,)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
SPACE_ID = os.environ.get('SPACE_ID', '')

DIY_MODE = ['Customized Mode 1: First A then B', 
            'Customized Mode 2: Both A and B', 
            'Customized Mode 3: RAW Camera Poses']


#### Description ####
title = r"""<h1 align="center">MotionCtrl: A Unified and Flexible Motion Controller for Video Generation</h1>"""
# subtitle = r"""<h2 align="center">Deployed on SVD Generation</h2>"""
important_link = r"""
<div align='center'>
<a href='https://huggingface.co/spaces/TencentARC/MotionCtrl_SVD'>[Demo MotionCtrl + SVD]</a>
&ensp; <a href='https://wzhouxiff.github.io/projects/MotionCtrl/assets/paper/MotionCtrl.pdf'>[Paper]</a>
&ensp; <a href='https://wzhouxiff.github.io/projects/MotionCtrl/'>[Project Page]</a>
&ensp; <a href='https://github.com/TencentARC/MotionCtrl'>[Code]</a>
&ensp; <a href='https://github.com/TencentARC/MotionCtrl/blob/svd/doc/showcase_svd.md'>[Showcases]</a>
&ensp; <a href='https://github.com/TencentARC/MotionCtrl/blob/svd/doc/tutorial.md'>[Tutorial]</a>
</div>
"""

description = r"""
<b>Official Gradio demo</b> for <a href='https://github.com/TencentARC/MotionCtrl' target='_blank'><b>MotionCtrl: A Unified and Flexible Motion Controller for Video Generation</b></a>.<br>
🔥 MotionCtrl is capable of independently and flexibly controling the camera motion and object motion of a generated video, with only a unified model.<br>
🤗 Try to control the motion of the generated videos yourself!<br>
❗❗❗ This demo provides model of **MotionCtrl** deployed on **LVDM/VideoCrafter** and **VideoCrafte2**. 
Deployments in **LVDM/VideoCrafter** include both Camera and Object Motion Control, 
while deployments in **VideoCrafte2** only include Camera Motion Control.
<br>
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
@inproceedings{wang2024motionctrl,
  title={Motionctrl: A unified and flexible motion controller for video generation},
  author={Wang, Zhouxia and Yuan, Ziyang and Wang, Xintao and Li, Yaowei and Chen, Tianshui and Xia, Menghan and Luo, Ping and Shan, Ying},
  booktitle={ACM SIGGRAPH 2024 Conference Papers},
  pages={1--11},
  year={2024}
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

exp_no = 0


for i in range(0, 16):
    # theta = (1)*np.pi*i/n

    R = base_R[:,:3]
    T = np.array([0.,0.,1.]).reshape(3,1) * (i/n)*2
    RT = np.concatenate([R,T], axis=1)
    res.append(RT)
    
fig = vis_camera(res)


###########################################

model_path='./checkpoints/motionctrl.pth'
config_path='./configs/inference/config_both.yaml'
if not os.path.exists(model_path):
    os.system(f'wget https://huggingface.co/TencentARC/MotionCtrl/resolve/main/motionctrl.pth?download=true -P ./checkpoints/')
    os.system(f'mv ./checkpoints/motionctrl.pth?download=true ./checkpoints/motionctrl.pth')

config = OmegaConf.load(config_path)
model_config = config.pop("model", OmegaConf.create())
model_v1 = instantiate_from_config(model_config)
if torch.cuda.is_available():
    model_v1 = model_v1.cuda()

model_v1 = load_model_checkpoint(model_v1, model_path)
model_v1.eval()

v2_model_path = './checkpoints/motionctrl_videocrafter2_cmcm.ckpt'
if not os.path.exists(v2_model_path):
    os.system(f'wget https://huggingface.co/TencentARC/MotionCtrl/resolve/main/motionctrl_videocrafter2_cmcm.ckpt?download=true -P ./checkpoints/')
    os.system(f'mv ./checkpoints/motionctrl_videocrafter2_cmcm.ckpt?download=true ./checkpoints/motionctrl_videocrafter2_cmcm.ckpt')

model_v2 = instantiate_from_config(model_config)
model_v2 = load_model_checkpoint(model_v2, v2_model_path)

if torch.cuda.is_available():
    model_v2 = model_v2.cuda()

model_v2.eval()


def model_run(prompts, choose_model, infer_mode, seed, n_samples, camera_args=None):
    traj_list = get_traj_list()
    camera_dict = get_camera_dict()

    RT = process_camera(camera_dict, camera_args).reshape(-1,12)
    traj_flow = process_traj(traj_list).transpose(3,0,1,2)

    if choose_model == BASE_MODEL[0]:
        model = model_v1
        noise_shape = [1, 4, 16, 32, 32]
    else:
        model = model_v2
        noise_shape = [1, 4, 16, 40, 64]
    unconditional_guidance_scale = 7.5
    unconditional_guidance_scale_temporal = None

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
    
    file_path = save_results(batch_variants, fps=10)

    return gr.update(value=file_path, width=256*n_samples, height=256)

    # return 

def save_results(video, fps=10, out_dir=None):
    
    # b,c,t,h,w
    video = video.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
    frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n)) for framesheet in video] #[3, 1*h, n*w]
    grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [t, h, w*n, 3]
    
    if out_dir is None:
        path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    else:
        path = os.path.join(out_dir, 'motionctrl.mp4')

    writer = imageio.get_writer(path, format='mp4', mode='I', fps=fps)
    for i in range(grid.shape[0]):
        img = grid[i].numpy()
        writer.append_data(img)

    writer.close()

    return path


def main(args):
    demo = gr.Blocks()
    with demo:

        gr.Markdown(title)
        gr.Markdown(important_link)
        gr.Markdown(description)


        with gr.Column():
            # step 0: select based model.
            gr.Markdown("## Step0: Selecting the model", show_label=False)
            gr.Markdown( f'- {BASE_MODEL[0]}: **MotionCtrl** deployed on {BASE_MODEL[0]}', show_label=False)
            gr.Markdown( f'- {BASE_MODEL[1]}: **MotionCtrl** deployed on {BASE_MODEL[1]}', show_label=False)
            # gr.HighlightedText(value=[("",""), (f'Choosing {BASE_MODEL[1]} requires time for loading new model. Please be patient.', "Normal")],
            #                     color_map={"Normal": "green", "Error": "red", "Clear clicks": "gray", "Add mask": "green", "Remove mask": "red"}, visible=True)
            choose_model = gr.Radio(choices=BASE_MODEL, value=BASE_MODEL[0], label="Based Model", interactive=True)
            choose_model_button = gr.Button(value="Proceed")

            # step 1: select motion control mode
            step1 = gr.Markdown("## Step 1/3: Selecting the motion control mode", show_label=False, visible=False)
            setp1_dec = gr.Markdown( f'\n - {MODE[0]}: Control the camera motion only \
                                       \n- {MODE[1]}: Control the object motion only \
                                       \n- {MODE[2]}: Control both the camera and object motion \
                                       \n- Click `Proceed` to go into next step',
                                       show_label=False, visible=False)
            infer_mode = gr.Radio(choices=MODE, value=MODE[0], label="Motion Control Mode", interactive=True, visible=False)
            mode_info = gr.Button(value="Proceed", visible=False)

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
                    # custom_run_status = gr.Markdown(f"\n 1. Click two of the basic camera poses, such as `Pan Up` and `Pan Left`; \
                    #                                 \n 2. Click `Customized Mode 1: First A then B` or `Customized Mode 1: First A then B` \
                    #                                 \n - `Customized Mode 1: First A then B`: The camera first `Pan Up` and then `Pan Left`; \
                    #                                 \n - `Customized Mode 2: Both A and B`: The camera move towards the upper left corner; \
                    #                                 \n 3. Slide the `Motion speed` to get a speed value. The large the value, the fast the camera motion; \
                    #                                 \n 4. Click `Visualize Camera and Proceed` to visualize the camera poses and go proceed; \
                    #                                 \n 5. Click `Reset Camera` to reset the camera poses (If needed). ",
                    #                                     show_label=False, visible=False)
                    custom_run_status = gr.Markdown(f"\n 1. Click `{DIY_MODE[0]}`, `{DIY_MODE[1]}`, or `{DIY_MODE[2]}` \
                                                    \n - `Customized Mode 1: First A then B`: For example, click `Pan Up` and `Pan Left`, the camera will first `Pan Up` and then `Pan Left`; \
                                                    \n - `Customized Mode 2: Both A and B`: For example, click `Pan Up` and `Pan Left`, the camera will move towards the upper left corner; \
                                                    \n - `{DIY_MODE[2]}`: Input the RAW RT matrix yourselves. \
                                                    \n 2. Slide the `Motion speed` to get a speed value. The large the value, the fast the camera motion; \
                                                    \n 3. Click `Visualize Camera and Proceed` to visualize the camera poses and go proceed; \
                                                    \n 4. Click `Reset Camera` to reset the camera poses (If needed). ",
                                                        show_label=False, visible=False)

                    # gr.HighlightedText(value=[("",""), ("1. Select two of the basic camera poses; 2. Select Customized Mode 1 OR Customized Mode 2. 3. Visualized Camera to show the customized camera poses", "Normal")],
                    #                                     color_map={"Normal": "green", "Error": "red", "Clear clicks": "gray", "Add mask": "green", "Remove mask": "red"}, visible=False)
                    
                    gr.HighlightedText(value=[("",""), ("1. Select two of the basic camera poses; 2. Select Customized Mode 1 OR Customized Mode 2. 3. Visualized Camera to show the customized camera poses", "Normal")],
                                                        color_map={"Normal": "green", "Error": "red", "Clear clicks": "gray", "Add mask": "green", "Remove mask": "red"}, visible=False)
                    
                    with gr.Row():
                        combine1 = gr.Button(value=DIY_MODE[0], visible=False)
                        combine2 = gr.Button(value=DIY_MODE[1], visible=False)
                        combine3 = gr.Button(value=DIY_MODE[2], visible=False)
                    with gr.Row():
                        combine3_des = gr.Markdown(f"---\n#### Input your camera pose in the following textbox. \
                                                A total of 14 lines and each line contains 12 float number, indicated \
                                                the RT matrix in the shape of 1x12. \
                                                The example is RT matrix of ZOOM IN.", show_label=False, visible=False)


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
                    
                    # with gr.Row():
                    #     combine1 = gr.Button(value="Customized Mode 1: First A then B", visible=False)
                    #     combine2 = gr.Button(value="Customized Mode 2: Both A and B", visible=False)

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
                    n_samples = gr.Number(value=2, precision=0, interactive=True, label="n_samples", visible=False)
                    seed = gr.Number(value=1234, precision=0, interactive=True, label="Seed", visible=False)
                    start = gr.Button(value="Start generation !", visible=False)
                with gr.Column():
                    gen_video = gr.Video(value=None, label="Generate Video", visible=False)

        choose_model_button.click(
            fn=visualized_step1,
            inputs=[choose_model],
            outputs=[
                     step1, setp1_dec, infer_mode, mode_info,
                     step2_camera_motion, 
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
                     combine1, combine2, combine3, combine3_des,
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
                     combine1, combine2, combine3, combine3_des,
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
                     combine1, combine2, combine3, combine3_des,
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

        combine1.click(fn=change_camera_mode, 
                       inputs=[combine1, camera_mode], 
                       outputs=[camera_args,
                                U, D, L, R, 
                                I, O, ACW, CW, speed,
                                combine3_des])
        combine2.click(fn=change_camera_mode, 
                       inputs=[combine2, camera_mode], 
                       outputs=[camera_args,
                                U, D, L, R, 
                                I, O, ACW, CW, speed,
                                combine3_des])
        combine3.click(fn=input_raw_camera_pose, 
                       inputs=[combine3, camera_mode], 
                       outputs=[camera_args,
                                U, D, L, R, 
                                I, O, ACW, CW, 
                                speed, 
                                combine3_des])

        camera_vis.click(fn=fn_vis_camera, inputs=[infer_mode, camera_args], outputs=[vis_camera, object_mode, object_info, step3_prompt_generate, prompt, n_samples, seed, start, gen_video])

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


        start.click(fn=model_run, inputs=[prompt, choose_model, infer_mode, seed, n_samples, camera_args], outputs=gen_video)

        gr.Markdown(article)

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

