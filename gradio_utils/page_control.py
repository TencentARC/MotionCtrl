import cv2
import numpy as np
import tempfile
import imageio
import gradio as gr
from gradio_utils.camera_utils import CAMERA_MOTION_MODE, process_camera, create_relative
from gradio_utils.traj_utils import get_provided_traj, process_points
from gradio_utils.utils import vis_camera

MODE = ["control camera poses", "control object trajectory", "control both camera and object motion"]

BASE_MODEL = ['LVDM/VideoCrafter', 'VideoCrafter2']

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

global traj_list, camera_dict

traj_list = [] 
camera_dict = {
                "motion":[],
                "mode": "Customized Mode 1: First A then B",  # "First A then B", "Both A and B", "Custom"
                "speed": 1.0,
                "complex": None
                }

def get_traj_list():
    global traj_list
    return traj_list

def get_camera_dict():
    global camera_dict
    return camera_dict

def reset_camera():
    global camera_dict
    camera_dict = {
                    "motion":[],
                    "mode": "Customized Mode 1: First A then B",
                    "speed": 1.0,
                    "complex": None
                    }   
    return display_camera_info(camera_dict)

def fn_traj_reset():
    global traj_list
    traj_list = []
    return "Click to specify trajectory"

def visualized_step1(model_name):

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
    vis_combine3 = False
    vis_combine3_des = False
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

    vis_step2_camera_motion = False
    vis_step2_camera_motion_des = False
    vis_camera_mode = False
    vis_camera_info = False

    vis_step2_object_motion = False
    vis_step2_object_motion_des = False
    vis_traj_mode = False
    vis_traj_info = False

    step2_camera_object_motion = False
    step2_camera_object_motion_des = False

    vis_step1 = True
    vis_step1_dec = True
    vis_infer_mode = True
    mode_info = True

    if model_name == BASE_MODEL[0]:
        interative_mode = True
    else:
        interative_mode = False
      
    return  gr.update(visible=vis_step1), \
            gr.update(visible=vis_step1_dec), \
            gr.update(visible=vis_infer_mode, value=MODE[0], interactive=interative_mode), \
            gr.update(visible=mode_info), \
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
            gr.update(visible=vis_combine1), gr.update(visible=vis_combine2), gr.update(visible=vis_combine3), gr.update(visible=vis_combine3_des), \
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
    vis_combine3 = False
    vis_combine3_des = False
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
        vis_step2_camera_motion = True
        vis_step2_camera_motion_des = True
        vis_camera_mode = True
        vis_camera_info = True
    
        vis_step2_object_motion = False
        vis_step2_object_motion_des = False
        vis_traj_mode = False
        vis_traj_info = False

        step2_camera_object_motion = True
        step2_camera_object_motion_des = True
      
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
            gr.update(visible=vis_combine1), gr.update(visible=vis_combine2), gr.update(visible=vis_combine3), gr.update(visible=vis_combine3_des), \
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
        vis_combine3_des = True
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
            gr.update(visible=vis_combine1), gr.update(visible=vis_combine2), gr.update(visible=vis_combine3), gr.update(visible=vis_combine3_des), \
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

    # return display_camera_info(camera_dict, camera_mode)
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

    return gr.update(value='1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 0 0 1 0 0 0 0 1 -0.225\n1 0 0 0 0 1 0 0 0 0 1 -0.45\n1 0 0 0 0 1 0 0 0 0 1 -0.675\n1 0 0 0 0 1 0 0 0 0 1 -0.9\n1 0 0 0 0 1 0 0 0 0 1 -1.125\n1 0 0 0 0 1 0 0 0 0 1 -1.35\n1 0 0 0 0 1 0 0 0 0 1 -1.575\n1 0 0 0 0 1 0 0 0 0 1 -1.8\n1 0 0 0 0 1 0 0 0 0 1 -2.025\n1 0 0 0 0 1 0 0 0 0 1 -2.25\n1 0 0 0 0 1 0 0 0 0 1 -2.475\n1 0 0 0 0 1 0 0 0 0 1 -2.7\n1 0 0 0 0 1 0 0 0 0 1 -2.925\n1 0 0 0 0 1 0 0 0 0 1 -3.15\n1 0 0 0 0 1 0 0 0 0 1 -3.375\n', max_lines=16, interactive=True), \
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

def add_traj_point(evt: gr.SelectData, ):
    global traj_list
    traj_list.append(evt.index)
    traj_str = [f"{traj}" for traj in traj_list]
    return ", ".join(traj_str)

def add_provided_traj(traj_name):
    global traj_list
    # import pdb; pdb.set_trace()
    traj_list = get_provided_traj(traj_name)
    traj_str = [f"{traj}" for traj in traj_list]
    return ", ".join(traj_str)


def fn_traj_droplast():
    global traj_list
    if traj_list:
        traj_list.pop()

    if traj_list:
        traj_str = [f"{traj}" for traj in traj_list]
        return ", ".join(traj_str)
    else:   
        return "Click to specify trajectory"

def fn_vis_camera(info_mode, camera_args=None):
    global camera_dict
    RT = process_camera(camera_dict, camera_args) # [t, 3, 4]

    rescale_T = 1.0
    rescale_T = max(rescale_T, np.max(np.abs(RT[:,:,-1])) / 1.9)

    fig = vis_camera(create_relative(RT), rescale_T=rescale_T)

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
    # import pdb; pdb.set_trace()
    # global traj_list
    # xy_range = 1024
    # print(traj_list)
    global traj_list
    print(traj_list)
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

    path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    writer = imageio.get_writer(path, format='mp4', mode='I', fps=10)
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

