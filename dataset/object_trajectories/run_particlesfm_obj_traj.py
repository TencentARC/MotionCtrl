# ParticleSfM
# Copyright (C) 2022  ByteDance Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Run the whole pipeline of trajectory-based video sfm from images
images -> optical flow -> point trajectories -> motion seg -> global mapper
"""
import argparse
import math
import os
import shutil
import time
from glob import glob

import torch
from motion_seg.core.network.traj_oa_depth import traj_oa_depth
from motion_seg.core.utils.utils import load_config_file
from point_trajectory import main_connect_point_trajectories
from third_party.MiDaS.midas.midas_net import MidasNet
from third_party.RAFT.compute_raft_custom_folder import infer_optical_flows
from third_party.RAFT.compute_raft_custom_folder_stride2 import \
    infer_optical_flows_stride2
from third_party.RAFT.core.raft import RAFT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.environ['TORCH_HOME'] = 'experiments/torch'

'''
- To solve the environment issue, I set the environment variables explicitly in the script.
- This setting requires to install the necessary libraries following our instructions.
- It may not need to be set in the script if the environment is set correctly in the system.
(Zhouxia Wang)
'''

colmap_cuda_path='envs/sfm_3d_colmap_cuda'
sfm_3d_path='envs/sfm_3d_root'

old_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
print(f'old_ld_library_path: {old_ld_library_path}')
os.environ['LD_LIBRARY_PATH'] = f'{colmap_cuda_path}/lib:{colmap_cuda_path}/lib64:{sfm_3d_path}/lib:{sfm_3d_path}/lib64:/usr/lib64:{old_ld_library_path}'
print(f'new_ld_library_path: {os.environ["LD_LIBRARY_PATH"]}')

old_path = os.environ.get('PATH', '')
print(f'old_path: {old_path}')
os.environ['PATH'] = f'{colmap_cuda_path}/bin:{sfm_3d_path}/bin:{old_path}'
print(f'new_path: {os.environ["PATH"]}')

old_include_path = os.environ.get('C_INCLUDE_PATH', '')
print(f'old_include_path: {old_include_path}')
os.environ['C_INCLUDE_PATH'] = f'{colmap_cuda_path}/include:{sfm_3d_path}/include:/usr/include:{old_include_path}'
print(f'new_include_path: {os.environ["C_INCLUDE_PATH"]}')

def connect_point_trajectory(args, image_dir, output_dir, skip_exists=False, keep_intermediate=False):
    # set directories in the workspace
    flow_dir = os.path.join(output_dir, "optical_flows")
    traj_dir = os.path.join(output_dir, "trajectories")

    print("[ParticleSFM] Running pairwise optical flow inference......")
    infer_optical_flows(args.raft_model, image_dir, flow_dir, skip_exists=skip_exists)

    if not args.skip_path_consistency:
        print("[ParticleSfM] Running pairwise optical flow inference (stride 2)......")
        infer_optical_flows_stride2(args.raft_model, image_dir, flow_dir, skip_exists=skip_exists)

    # point trajectory (saved in workspace_dir / point_trajectories)
    print("[ParticleSfM] Connecting (optimization {0}) point trajectories from optical flows.......".format("disabled" if args.skip_path_consistency else "enabled"))
    main_connect_point_trajectories(flow_dir, traj_dir, sample_ratio=args.sample_ratio, flow_check_thres=args.flow_check_thres, skip_path_consistency=args.skip_path_consistency, skip_exists=skip_exists)

    if not keep_intermediate:
        # remove optical flows
        shutil.rmtree(os.path.join(output_dir, "optical_flows"))
    return traj_dir

def motion_segmentation(args, image_dir, output_dir, traj_dir, skip_exists=False, keep_intermediate=False):
    # set directories in the workspace
    depth_dir = os.path.join(output_dir, "midas_depth")
    labeled_traj_dir = traj_dir + "_labeled"

    # monocular depth (MiDaS)
    print("[ParticleSfM] Running per-frame monocular depth estimation........")
    from third_party.MiDaS import run_midas_v21
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    run_midas_v21(args.midas_model, DEVICE, image_dir, depth_dir, skip_exists=skip_exists)

    # point trajectory based motion segmentation
    print("[ParticleSfM] Running point trajectory based motion segmentation........")
    from motion_seg.main_motion_segmentation import main_motion_segmentation_v1
    main_motion_segmentation_v1(args.motion_seg_model, args.motion_seg_cfg, 
                                image_dir, depth_dir, traj_dir, labeled_traj_dir, 
                                window_size=args.window_size, traj_max_num=args.traj_max_num, 
                                skip_exists=skip_exists)
    if os.path.isfile(os.path.join(output_dir, "motion_seg.mp4")):
        os.remove(os.path.join(output_dir, "motion_seg.mp4"))
    shutil.move(os.path.join(labeled_traj_dir, "motion_seg.mp4"), output_dir)

    if not keep_intermediate:
        # remove original point trajectories
        shutil.rmtree(depth_dir)
        shutil.rmtree(traj_dir)
    return labeled_traj_dir

def sfm_reconstruction(args, image_dir, output_dir, traj_dir, skip_exists=False, keep_intermediate=False):
    # set directories in the workspace
    sfm_dir = os.path.join(output_dir, "sfm")

    # sfm reconstruction
    from sfm import (main_global_sfm, main_incremental_sfm,
                     write_depth_pose_from_colmap_format)
    if not args.incremental_sfm:
        print("[ParticleSfM] Running global structure-from-motion........")
        main_global_sfm(sfm_dir, image_dir, traj_dir, colmap_path='envs/sfm_3d_colmap_cuda/bin/colmap', remove_dynamic=(not args.assume_static), skip_exists=skip_exists)
    else:
        print("[ParticleSfM] Running incremental structure-from-motion with COLMAP........")
        main_incremental_sfm(sfm_dir, image_dir, traj_dir, remove_dynamic=(not args.assume_static), skip_exists=skip_exists)

    # # write depth and pose files from COLMAP format
    write_depth_pose_from_colmap_format(sfm_dir, os.path.join(output_dir, "colmap_outputs_converted"))

    # if not keep_intermediate:
    #     # remove labeled point trajectories
    #     shutil.rmtree(traj_dir)

def particlesfm(args, image_dir, output_dir, skip_exists=False, keep_intermediate=False):
    """
    Inputs:
    - img_dir: str - The folder containing input images
    - output_dir: str - The workspace directory
    """
    if not os.path.exists(image_dir):
        raise ValueError("Error! The input image directory {0} is not found.".format(image_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # import pdb
    # pdb.set_trace()
    # connect point trajectory
    traj_dir = os.path.join(output_dir, "trajectories")
    traj_dir = connect_point_trajectory(args, image_dir, output_dir, skip_exists=skip_exists, keep_intermediate=keep_intermediate)

    # motion segmentation
    # traj_dir = traj_dir + "_labeled"
    # if not args.assume_static and not os.path.exists(traj_dir):
    if not args.assume_static:
        traj_dir = motion_segmentation(args, image_dir, output_dir, traj_dir, skip_exists=skip_exists, keep_intermediate=keep_intermediate)

    # sfm reconstruction
    # if not args.skip_sfm:
    #     sfm_reconstruction(args, image_dir, output_dir, traj_dir, skip_exists=skip_exists, keep_intermediate=keep_intermediate)

def parse_args():
    parser = argparse.ArgumentParser("Dense point trajectory based colmap reconstruction for videos")
    # point trajectory
    parser.add_argument("--flow_check_thres", type=float, default=1.0, help='the forward-backward flow consistency check threshold')
    parser.add_argument("--sample_ratio", type=int, default=2, help='the sampling ratio for point trajectories')
    parser.add_argument("--traj_min_len", type=int, default=3, help='the minimum length for point trajectories')
    # motion segmentation
    parser.add_argument("--window_size", type=int, default=10, help='the window size for trajectory motion segmentation')
    parser.add_argument("--traj_max_num", type=int, default=100000, help='the maximum number of trajs inside a window')
    # sfm
    parser.add_argument("--incremental_sfm", action='store_true', help='whether to use incremental sfm or not')
    # pipeline control
    parser.add_argument("--skip_path_consistency", action='store_true', help='whether to skip the path consistency optimization or not')
    parser.add_argument("--assume_static", action='store_true', help='whether to skip the motion segmentation or not')
    parser.add_argument("--skip_sfm", action='store_true', help='whether to skip structure-from-motion or not')
    parser.add_argument("--skip_exists", action='store_true', help='whether to skip exists')
    parser.add_argument("--keep_intermediate", action='store_true', help='whether to keep intermediate files such as flows, monocular depths, etc.')

    # input by sequence directory
    # python run_particlesfm.py --image_dir ${PATH_TO_SEQ_FOLDER} --output_dir ${OUTPUT_WORKSPACE}
    parser.add_argument("-i", "--image_dir", type=str, default="none", help="path to the sequence folder containing images")
    parser.add_argument("-o", "--output_dir", type=str, default="none", help="workspace for output")

    # input by workspace
    # python run_particlesfm.py --workspace_dir ${WORKSPACE_DIR}
    parser.add_argument("--workspace_dir", type=str, default="none", help="input workspace")
    parser.add_argument("--image_folder", type=str, default="images", help="image folder") # also used in the folder option
    parser.add_argument("--start_idx", type=int, default=0, help='the starting index of the sequence')
    parser.add_argument("--end_idx", type=int, default=math.inf, help='the ending index of the sequence')

    # input by folder containing multiple workspaces
    # python run_particlesfm.py --root_dir ${ROOT_DIR}
    # multiple sequences should be with the structure below:
    # - ROOT_DIR
    #    - XXX (sequence 1)
    #        - images
    #            - xxxxxx.png
    #    - XXX (sequence 2)
    parser.add_argument("--root_dir", type=str, default="none", help='path to to the folder containing workspaces')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # --------------------------------------------------------------------------
    # load RAFT model
    raft_args = argparse.Namespace()
    raft_args.small = False
    raft_args.alternate_corr = False
    raft_args.mixed_precision = False
    raft_args.model = 'third_party/RAFT/models/raft-things.pth'

    raft_model = RAFT(raft_args)

    weights = torch.load(raft_args.model, map_location='cpu')
    new_weights = {}
    for k, v in weights.items():
        new_weights[k.replace('module.', '')] = v
    raft_model.load_state_dict(new_weights)
    print("Loaded successfully: {0}".format(raft_args.model))

    raft_model.to(DEVICE)
    raft_model.eval()

    args.raft_model = raft_model

    # --------------------------------------------------------------------------
    # load motion segmentation model
    config_file = "motion_seg/configs/example_test.yaml"

    cfg = load_config_file(config_file)
    if cfg.model_name == "traj_oa_depth":
        motion_seg_model = traj_oa_depth(args.window_size, cfg.resolution)
    else:
        raise NotImplementedError
    if cfg.resume_path:
        resume_path = os.path.join('motion_seg', cfg.resume_path)
        checkpoint = torch.load(resume_path)
        motion_seg_model.load_state_dict(checkpoint['model_state_dict'])
        print('Load motion segmentation model from {}'.format(resume_path))
    else:
        raise NotImplementedError

    motion_seg_model.to(DEVICE)
    motion_seg_model.eval()

    args.motion_seg_model = motion_seg_model
    args.motion_seg_cfg = cfg

    # --------------------------------------------------------------------------
    # load MiDaS model
    model_path = "third_party/MiDaS/weights/midas_v21-f6b98070.pt"
    midas_model = MidasNet(model_path, non_negative=True)
    midas_model.eval()
    # force to optimize=True
    if DEVICE == torch.device("cuda"):
        midas_model.to(memory_format=torch.channels_last)
        midas_model.half()
    midas_model.to(DEVICE)
    args.midas_model = midas_model

    # --------------------------------------------------------------------------
    # run particlesfm

    if args.image_dir != "none" and args.output_dir != "none": # input by sequence directory
        particlesfm(args, args.image_dir, args.output_dir, skip_exists=args.skip_exists, keep_intermediate=args.keep_intermediate)
    elif args.workspace_dir != "none":
        image_dir = os.path.join(args.workspace_dir, args.image_folder)
        particlesfm(args, image_dir, args.workspace_dir, skip_exists=args.skip_exists, keep_intermediate=args.keep_intermediate)
    elif args.root_dir != "none":

        if not os.path.exists(args.root_dir):
            raise ValueError("Error! The input folder {0} is not found.".format(args.root_dir))
        # seq_names = sorted(os.listdir(args.root_dir))
        seq_names = sorted(glob(os.path.join(args.root_dir, '*', '*')))
        print("A total of {0} sequences found in {1}.".format(len(seq_names), args.root_dir))

        if args.start_idx < 0:
            args.start_idx = 0
        if args.end_idx > len(seq_names):
            args.end_idx = len(seq_names)
        print('*'*80)
        print("Running sequences from {0} to {1}.".format(args.start_idx, args.end_idx))
        print('*'*80)

        # for seq_name in seq_names[args.start_idx:args.end_idx]:
        for i in range(args.start_idx, args.end_idx):
            seq_name = seq_names[i]
            start_time = time.time()
            # workspace_dir = os.path.join(args.root_dir, seq_name)
            workspace_dir = seq_name
            image_dir = os.path.join(workspace_dir, args.image_folder)
            traj_dir = os.path.join(workspace_dir, 'trajectories_labeled/track.npy')
            if os.path.exists(traj_dir):
                print(f'!!!!!! Skipping {seq_name} because it has been processed.')
                continue
            # particlesfm(args, image_dir, workspace_dir, skip_exists=args.skip_exists, keep_intermediate=args.keep_intermediate)
            # print("Sequence {0} finished in {1:.2f} seconds.".format(seq_name, time.time() - start_time))
            # print('-' * 80)
            try:
                particlesfm(args, image_dir, workspace_dir, skip_exists=args.skip_exists, keep_intermediate=args.keep_intermediate)
                print("Sequence {0} finished in {1:.2f} seconds.".format(seq_name, time.time() - start_time))
                print('-' * 80)
            except:
                print("Sequence {0} failed.".format(seq_name))
                print('-' * 80)
            print(f'Finished {i+1}/{args.end_idx-args.start_idx} sequences.')
    else:       
        raise ValueError("Error! No input provided.")