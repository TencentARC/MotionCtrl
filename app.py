import argparse
import os
import tempfile

import gradio as gr
import numpy as np
import torch
from glob import glob
from torchvision.transforms import CenterCrop, Compose, Resize

from gradio_utils.camera_utils import CAMERA_MOTION_MODE, process_camera, create_relative

from gradio_utils.utils import vis_camera
from gradio_utils.motionctrl_cmcm_gradio import build_model, motionctrl_sample

os.environ['KMP_DUPLICATE_LIB_OK']='True'
SPACE_ID = os.environ.get('SPACE_ID', '')


#### Description ####
title = r"""<h1 align="center">MotionCtrl: A Unified and Flexible Motion Controller for Video Generation</h1>"""
subtitle = r"""<h2 align="center">Deployed on SVD Generation</h2>"""
important_link = r"""
<div align='center'>
<a href='https://wzhouxiff.github.io/projects/MotionCtrl/assets/paper/MotionCtrl.pdf'>[Paper]</a>
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
❗❗❗ Please note **ONLY** Camera Motion Control in the current version of **MotionCtrl** deployed on **SVD** is avaliable.<br>
❗❗❗ <a href='https://github.com/TencentARC/MotionCtrl/blob/svd/doc/showcase_svd.md' target='_blank'>Showcases</a> and 
<a href='https://github.com/TencentARC/MotionCtrl/blob/svd/doc/tutorial.md' target='_blank'>Tutorial</a> can be found 
<a href='https://github.com/TencentARC/MotionCtrl/blob/svd/doc/tutorial.md' target='_blank'>here</a><br>.
"""
# <div>
# <img src="https://raw.githubusercontent.com/TencentARC/MotionCtrl/main/assets/svd/00_ibzz5-dxv2h.gif", width="300">
# <img src="https://raw.githubusercontent.com/TencentARC/MotionCtrl/main/assets/svd/01_5guvn-0x6v2.gif", width="300">
# <img src="https://raw.githubusercontent.com/TencentARC/MotionCtrl/main/assets/svd/12_sn7bz-0hcaf.gif", width="300">
# <img src="https://raw.githubusercontent.com/TencentARC/MotionCtrl/main/assets/svd/13_3lyco-4ru8j.gif", width="300">
# </div>
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
            [1.,0.,0.],             ## W2C  x 的正方向： 相机朝左  left
            [-1.,0.,0.],            ## W2C  x 的负方向： 相机朝右  right
            [0., 1., 0.],           ## W2C  y 的正方向： 相机朝上  up     
            [0.,-1.,0.],            ## W2C  y 的负方向： 相机朝下  down
            [0.,0.,1.],             ## W2C  z 的正方向： 相机往前  zoom out
            [0.,0.,-1.],            ## W2C  z 的负方向： 相机往前  zoom in
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
RESIZE_MODE = ['Center Crop To 576x1024', 'Keep original spatial ratio']
DIY_MODE = ['Customized Mode 1: First A then B', 
            'Customized Mode 2: Both A and B', 
            'Customized Mode 3: RAW Camera Poses']

## load default model
num_frames = 14
num_steps = 25
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

config = "configs/inference/config_motionctrl_cmcm.yaml"
ckpt='checkpoints/motionctrl_svd.ckpt'
if not os.path.exists(ckpt):
    os.system(f'wget https://huggingface.co/TencentARC/MotionCtrl/resolve/main/motionctrl_svd.ckpt?download=true -P .')
    os.system(f'mkdir checkpoints')
    os.system(f'mv motionctrl_svd.ckpt?download=true {ckpt}')
model = build_model(config, ckpt, device, num_frames, num_steps)
width, height = 1024, 576 

traj_list = [] 
camera_dict = {
                "motion":[],
                "mode": "Customized Mode 1: First A then B",  # "First A then B", "Both A and B", "Custom"
                "speed": 1.0,
                "complex": None
                }   

def fn_vis_camera(camera_args):
    global camera_dict, num_frames, width, height
    RT = process_camera(camera_dict, camera_args, num_frames=num_frames, width=width, height=height) # [t, 3, 4]

    rescale_T = 1.0
    rescale_T = max(rescale_T, np.max(np.abs(RT[:,:,-1])) / 1.9)

    fig = vis_camera(create_relative(RT), rescale_T=rescale_T)

    vis_step3_prompt_generate = True
    vis_generation_dec = True
    vis_prompt = True
    vis_num_samples = True
    vis_seed = True
    vis_start = True
    vis_gen_video = True
    vis_repeat_highlight = True

    return fig, \
            gr.update(visible=vis_step3_prompt_generate), \
            gr.update(visible=vis_generation_dec), \
            gr.update(visible=vis_prompt), \
            gr.update(visible=vis_num_samples), \
            gr.update(visible=vis_seed), \
            gr.update(visible=vis_start), \
            gr.update(visible=vis_gen_video, value=None), \
            gr.update(visible=vis_repeat_highlight)

def display_camera_info(camera_dict, camera_mode=None):
    if camera_dict['complex'] is not None:
        res = f"complex : {camera_dict['complex']}. "
        res += f"speed : {camera_dict['speed']}. "
    else:
        res = ""
        res += f"motion : {[_ for _ in camera_dict['motion']]}. "
        res += f"speed : {camera_dict['speed']}. "
        if camera_mode == CAMERA_MOTION_MODE[2]:
            res += f"mode : {camera_dict['mode']}. "
    return res

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

def input_raw_camera_pose(combine_type, camera_mode):
    global camera_dict
    camera_dict['mode'] = combine_type

    vis_U = False
    vis_D = False
    vis_L = False
    vis_R = False
    vis_I = False
    vis_O = False
    vis_ACW = False
    vis_CW = False
    vis_speed = True
    vis_combine3_des = True

    return gr.update(value='1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 0 0 1 0 0 0 0 1 -0.225\n1 0 0 0 0 1 0 0 0 0 1 -0.45\n1 0 0 0 0 1 0 0 0 0 1 -0.675\n1 0 0 0 0 1 0 0 0 0 1 -0.9\n1 0 0 0 0 1 0 0 0 0 1 -1.125\n1 0 0 0 0 1 0 0 0 0 1 -1.35\n1 0 0 0 0 1 0 0 0 0 1 -1.575\n1 0 0 0 0 1 0 0 0 0 1 -1.8\n1 0 0 0 0 1 0 0 0 0 1 -2.025\n1 0 0 0 0 1 0 0 0 0 1 -2.25\n1 0 0 0 0 1 0 0 0 0 1 -2.475\n1 0 0 0 0 1 0 0 0 0 1 -2.7\n1 0 0 0 0 1 0 0 0 0 1 -2.925\n', max_lines=16, interactive=True), \
            gr.update(visible=vis_U), \
            gr.update(visible=vis_D), \
            gr.update(visible=vis_L),\
            gr.update(visible=vis_R), \
            gr.update(visible=vis_I), \
            gr.update(visible=vis_O), \
            gr.update(visible=vis_ACW), \
            gr.update(visible=vis_CW), \
            gr.update(visible=vis_speed), \
            gr.update(visible=vis_combine3_des)

def change_camera_mode(combine_type, camera_mode):
    global camera_dict
    camera_dict['mode'] = combine_type

    vis_U = True
    vis_D = True
    vis_L = True
    vis_R = True
    vis_I = True
    vis_O = True
    vis_ACW = True
    vis_CW = True
    vis_speed = True
    vis_combine3_des = False

    return display_camera_info(camera_dict, camera_mode), \
            gr.update(visible=vis_U), \
            gr.update(visible=vis_D), \
            gr.update(visible=vis_L),\
            gr.update(visible=vis_R), \
            gr.update(visible=vis_I), \
            gr.update(visible=vis_O), \
            gr.update(visible=vis_ACW), \
            gr.update(visible=vis_CW), \
            gr.update(visible=vis_speed), \
            gr.update(visible=vis_combine3_des)

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


def visualized_camera_poses(step2_camera_motion):
    reset_camera()

    # generate video
    vis_step3_prompt_generate = False
    vis_generation_dec = False
    vis_prompt = False
    vis_num_samples = False
    vis_seed = False
    vis_start = False
    vis_gen_video = False
    vis_repeat_highlight = False

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
        vis_combine3 = False
        vis_combine3_des = False
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
        vis_combine3 = False
        vis_combine3_des = False
        vis_speed = True

        vis_Pose_1, vis_Pose_2, vis_Pose_3, vis_Pose_4 = True, True, True, True
        vis_Pose_5, vis_Pose_6, vis_Pose_7, vis_Pose_8 = True, True, True, True

    else: # step2_camera_motion = CAMERA_MOTION_MODE[2]:
        vis_basic_camera_motion = False
        vis_basic_camera_motion_des = False
        vis_custom_camera_motion = True
        vis_custom_run_status = True
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
        vis_combine1 = True
        vis_combine2 = True
        vis_combine3 = True
        vis_combine3_des = False
        vis_speed = False

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
            gr.update(visible=vis_combine1), gr.update(visible=vis_combine2), gr.update(visible=vis_combine3), \
            gr.update(visible=vis_combine3_des), \
            gr.update(visible=vis_speed), \
            gr.update(visible=vis_Pose_1), gr.update(visible=vis_Pose_2), gr.update(visible=vis_Pose_3), gr.update(visible=vis_Pose_4), \
            gr.update(visible=vis_Pose_5), gr.update(visible=vis_Pose_6), gr.update(visible=vis_Pose_7), gr.update(visible=vis_Pose_8), \
            gr.update(visible=vis_camera_args, value=None), \
            gr.update(visible=vis_camera_reset), gr.update(visible=vis_camera_vis), \
            gr.update(visible=vis_vis_camera, value=None), \
            gr.update(visible=vis_step3_prompt_generate), \
            gr.update(visible=vis_generation_dec), \
            gr.update(visible=vis_prompt), \
            gr.update(visible=vis_num_samples), \
            gr.update(visible=vis_seed), \
            gr.update(visible=vis_start), \
            gr.update(visible=vis_gen_video), \
            gr.update(visible=vis_repeat_highlight)


def process_input_image(input_image, resize_mode):
    global width, height
    if resize_mode == RESIZE_MODE[0]:
        height = 576
        width = 1024
        w, h = input_image.size
        h_ratio = h / height
        w_ratio = w / width

        if h_ratio > w_ratio:
            h = int(h / w_ratio)
            if h < height:
                h = height
            input_image = Resize((h, width))(input_image)
            
        else:
            w = int(w / h_ratio)
            if w < width:
                w = width
            input_image = Resize((height, w))(input_image)

        transformer = Compose([
            # Resize(width),
            CenterCrop((height, width)),
        ])

        input_image = transformer(input_image)
    else:
        w, h = input_image.size
        if h > w:
            height = 576
            width = int(w * height / h)
        else:
            width = 1024
            height = int(h * width / w)

        input_image = Resize((height, width))(input_image)
        # print(f'input_image size: {input_image.size}')

    vis_step2_camera_motion = True
    vis_step2_camera_motion_des = True
    vis_camera_mode = True
    vis_camera_info = True

    ####
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
    vis_combine3 = False
    vis_combine3_des = False
    vis_speed = False

    vis_Pose_1, vis_Pose_2, vis_Pose_3, vis_Pose_4 = False, False, False, False
    vis_Pose_5, vis_Pose_6, vis_Pose_7, vis_Pose_8 = False, False, False, False

    vis_camera_args = False
    vis_camera_reset = False
    vis_camera_vis = False
    vis_vis_camera = False

    # generate video
    vis_step3_prompt_generate = False
    vis_generation_dec = False
    vis_prompt = False
    vis_num_samples = False
    vis_seed = False
    vis_start = False
    vis_gen_video = False
    vis_repeat_highlight = False
    
    return gr.update(visible=True, value=input_image, height=height, width=width), \
            gr.update(visible=vis_step2_camera_motion), \
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
            gr.update(visible=vis_combine1), gr.update(visible=vis_combine2), gr.update(visible=vis_combine3), \
            gr.update(visible=vis_combine3_des), \
            gr.update(visible=vis_speed), \
            gr.update(visible=vis_Pose_1), gr.update(visible=vis_Pose_2), gr.update(visible=vis_Pose_3), gr.update(visible=vis_Pose_4), \
            gr.update(visible=vis_Pose_5), gr.update(visible=vis_Pose_6), gr.update(visible=vis_Pose_7), gr.update(visible=vis_Pose_8), \
            gr.update(visible=vis_camera_args, value=None), \
            gr.update(visible=vis_camera_reset), gr.update(visible=vis_camera_vis), \
            gr.update(visible=vis_vis_camera, value=None), \
            gr.update(visible=vis_step3_prompt_generate), \
            gr.update(visible=vis_generation_dec), \
            gr.update(visible=vis_prompt), \
            gr.update(visible=vis_num_samples), \
            gr.update(visible=vis_seed), \
            gr.update(visible=vis_start), \
            gr.update(visible=vis_gen_video), \
            gr.update(visible=vis_repeat_highlight)

def model_run(input_image, fps_id, seed, n_samples, camera_args):
    global model, device, camera_dict, num_frames, num_steps, width, height
    RT = process_camera(camera_dict, camera_args, num_frames=num_frames, width=width, height=height).reshape(-1,12)

    video_path = motionctrl_sample(
        model=model,
        image=input_image,
        RT=RT,
        num_frames=num_frames,
        fps_id=fps_id,
        decoding_t=1,
        seed=seed,
        sample_num=n_samples,
        device=device
    )

    return video_path

def main(args):
    demo = gr.Blocks()
    with demo:

        gr.Markdown(title)
        gr.Markdown(subtitle)
        gr.Markdown(important_link)
        gr.Markdown(description)

        with gr.Column():
            
            # step 0: Some useful tricks
            gr.Markdown("## Step 0/3: Some Useful Tricks", show_label=False)
            gr.HighlightedText(value=[("",""), (f"1. If the motion control is not obvious, try to increase the `Motion Speed`. \
                                                \n 2. If the generated videos are distored severely, try to descrease the `Motion Speed` \
                                                or increase `FPS`.", "Normal")],
                                color_map={"Normal": "green", "Error": "red", "Clear clicks": "gray", "Add mask": "green", "Remove mask": "red"}, visible=True)

            # step 2: input an image
            step2_title = gr.Markdown("---\n## Step 1/3: Input an Image", show_label=False, visible=True)
            step2_dec = gr.Markdown(f"\n 1. Upload an Image by `Drag` or Click `Upload Image`; \
                                    \n 2. Click `{RESIZE_MODE[0]}` or `{RESIZE_MODE[1]}` to select the image resize mode. \
                                    You will get a processed image and go into the next step. \
                                    \n - `{RESIZE_MODE[0]}`: Our MotionCtrl is train on image with spatial size 576x1024. Choose `{RESIZE_MODE[0]}` can get better generated video. \
                                    \n - `{RESIZE_MODE[1]}`: Choose `{RESIZE_MODE[1]}` if you want to generate video with the same spatial ratio as the input image.", 
                                    show_label=False, visible=True)
                                    
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    input_image = gr.Image(type="pil", interactive=True, elem_id="input_image", elem_classes='image', visible=True)
                    # process_input_image_button = gr.Button(value="Process Input Image", visible=False)
                    with gr.Row():
                        center_crop_botton = gr.Button(value=RESIZE_MODE[0], visible=True)
                        keep_spatial_raition_botton = gr.Button(value=RESIZE_MODE[1], visible=True)
                with gr.Column(scale=2):
                    process_image = gr.Image(type="pil", interactive=False, elem_id="process_image", elem_classes='image', visible=False)
            # step2_proceed_button = gr.Button(value="Proceed", visible=False)

            
            # step3 - camera motion control
            step2_camera_motion = gr.Markdown("---\n## Step 2/3: Select the camera poses", show_label=False, visible=False)
            step2_camera_motion_des = gr.Markdown(f"\n - {CAMERA_MOTION_MODE[0]}: Including 8 basic camera poses, such as pan up, pan down, zoom in, and zoom out. \
                                                    \n - {CAMERA_MOTION_MODE[1]}: Complex camera poses extracted from the real videos. \
                                                    \n - {CAMERA_MOTION_MODE[2]}: You can customize complex camera poses yourself by combining or fusing two of the eight basic camera poses or input RAW RT matrix. \
                                                    \n - Click `Proceed` to go into next step", 
                                                  show_label=False, visible=False)
            camera_mode = gr.Radio(choices=CAMERA_MOTION_MODE, value=CAMERA_MOTION_MODE[0], label="Camera Motion Control Mode", interactive=True, visible=False)
            camera_info = gr.Button(value="Proceed", visible=False)

            with gr.Row():
                with gr.Column():
                    # step3.1 - camera motion control - basic
                    basic_camera_motion = gr.Markdown("---\n### Basic Camera Poses", show_label=False, visible=False)
                    basic_camera_motion_des = gr.Markdown(f"\n 1. Click one of the basic camera poses, such as `Pan Up`; \
                                                            \n 2. Slide the `Motion speed` to get a speed value. The large the value, the fast the camera motion; \
                                                            \n 3. Click `Visualize Camera and Proceed` to visualize the camera poses and go proceed; \
                                                            \n 4. Click `Reset Camera` to reset the camera poses (If needed). ",
                                                        show_label=False, visible=False)
                    
                    
                    # step3.2 - camera motion control - provided complex
                    complex_camera_motion = gr.Markdown("---\n### Provided Complex Camera Poses", show_label=False, visible=False)
                    complex_camera_motion_des = gr.Markdown(f"\n 1. Click one of the complex camera poses, such as `Pose_1`; \
                                                            \n 2. Click `Visualize Camera and Proceed` to visualize the camera poses and go proceed; \
                                                            \n 3. Click `Reset Camera` to reset the camera poses (If needed). ",
                                                        show_label=False, visible=False)

                    # step3.3 - camera motion control - custom
                    custom_camera_motion = gr.Markdown(f"---\n### {CAMERA_MOTION_MODE[2]}", show_label=False, visible=False)
                    custom_run_status = gr.Markdown(f"\n 1. Click `{DIY_MODE[0]}`, `{DIY_MODE[1]}`, or `{DIY_MODE[2]}` \
                                                    \n - `Customized Mode 1: First A then B`: For example, click `Pan Up` and `Pan Left`, the camera will first `Pan Up` and then `Pan Left`; \
                                                    \n - `Customized Mode 2: Both A and B`: For example, click `Pan Up` and `Pan Left`, the camera will move towards the upper left corner; \
                                                    \n - `{DIY_MODE[2]}`: Input the RAW RT matrix yourselves. \
                                                    \n 2. Slide the `Motion speed` to get a speed value. The large the value, the fast the camera motion; \
                                                    \n 3. Click `Visualize Camera and Proceed` to visualize the camera poses and go proceed; \
                                                    \n 4. Click `Reset Camera` to reset the camera poses (If needed). ",
                                                        show_label=False, visible=False)

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

                    with gr.Row():    
                        speed = gr.Slider(minimum=0, maximum=8, step=0.2, label="Motion Speed", value=1.0, visible=False)

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

            
            # step4 - Generate videos
            with gr.Row():
                with gr.Column():
                    step3_prompt_generate = gr.Markdown("---\n## Step 3/3: Generate videos", show_label=False, visible=False)
                    generation_dec = gr.Markdown(f"\n 1. Set `FPS`.; \
                                                    \n 2. Set `n_samples`; \
                                                    \n 3. Set `seed`; \
                                                    \n 4. Click `Start generation !` to generate videos; ", visible=False)
                    # prompt = gr.Textbox(value="a dog sitting on grass", label="Prompt", interactive=True, visible=False)
                    prompt = gr.Slider(minimum=5, maximum=30, step=1, label="FPS", value=10, visible=False)
                    n_samples = gr.Number(value=2, precision=0, interactive=True, label="n_samples", visible=False)
                    seed = gr.Number(value=1234, precision=0, interactive=True, label="Seed", visible=False)
                    start = gr.Button(value="Start generation !", visible=False)
                with gr.Column():
                    gen_video = gr.Video(value=None, label="Generate Video", visible=False)
                    repeat_highlight=gr.HighlightedText(value=[("",""), (f"1. If the motion control is not obvious, try to increase the `Motion Speed`. \
                                                \n 2. If the generated videos are distored severely, try to descrease the `Motion Speed` \
                                                or increase `FPS`.", "Normal")],
                                color_map={"Normal": "green", "Error": "red", "Clear clicks": "gray", "Add mask": "green", "Remove mask": "red"}, visible=False)

        center_crop_botton.click(
            fn=process_input_image, 
            inputs=[input_image, center_crop_botton], 
            outputs=[
                process_image,
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

                step3_prompt_generate, 
                generation_dec, 
                prompt, 
                n_samples, 
                seed, start, gen_video, repeat_highlight])

        keep_spatial_raition_botton.click(
            fn=process_input_image, 
            inputs=[input_image, keep_spatial_raition_botton], 
            outputs=[
                process_image,
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

                step3_prompt_generate, 
                generation_dec, 
                prompt, 
                n_samples, 
                seed, start, gen_video, repeat_highlight])
        

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
                     step3_prompt_generate, generation_dec, prompt, n_samples, seed, start, gen_video, repeat_highlight],
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
                                I, O, ACW, CW, 
                                speed,
                                combine3_des])
        combine3.click(fn=input_raw_camera_pose, 
                       inputs=[combine3, camera_mode], 
                       outputs=[camera_args,
                                U, D, L, R, 
                                I, O, ACW, CW, 
                                speed, 
                                combine3_des])

        camera_vis.click(fn=fn_vis_camera, inputs=[camera_args], 
                         outputs=[vis_camera, 
                                  step3_prompt_generate, 
                                  generation_dec,
                                  prompt, 
                                  n_samples, 
                                  seed, 
                                  start, 
                                  gen_video,
                                  repeat_highlight])

        Pose_1.click(fn=add_complex_camera_motion, inputs=Pose_1, outputs=camera_args)
        Pose_2.click(fn=add_complex_camera_motion, inputs=Pose_2, outputs=camera_args)
        Pose_3.click(fn=add_complex_camera_motion, inputs=Pose_3, outputs=camera_args)
        Pose_4.click(fn=add_complex_camera_motion, inputs=Pose_4, outputs=camera_args)
        Pose_5.click(fn=add_complex_camera_motion, inputs=Pose_5, outputs=camera_args)
        Pose_6.click(fn=add_complex_camera_motion, inputs=Pose_6, outputs=camera_args)
        Pose_7.click(fn=add_complex_camera_motion, inputs=Pose_7, outputs=camera_args)
        Pose_8.click(fn=add_complex_camera_motion, inputs=Pose_8, outputs=camera_args)


        start.click(fn=model_run, 
                    inputs=[process_image, prompt, seed, n_samples, camera_args], 
                    outputs=gen_video)

        # set example
        gr.Markdown("## Examples")
        examples = glob(os.path.join(os.path.dirname(__file__), "./assets/demo/images", "*.png"))
        gr.Examples(
            examples=examples,
            inputs=[input_image],
        )

        gr.Markdown(article)

    # demo.launch(server_name='0.0.0.0', share=False, server_port=args['server_port'])
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
