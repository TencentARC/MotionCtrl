
import math
import os
import random
import sys

import numpy as np
import omegaconf
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from motionctrl.data.utils import create_relative


def make_spatial_transformations(resolution, type, ori_resolution=None):
    """ 
    resolution: target resolution, a list of int, [h, w]
    """
    if type == "random_crop":
        transformations = transforms.RandomCropss(resolution)
    elif type == "resize_center_crop":
        is_square = (resolution[0] == resolution[1])
        if is_square:
            transformations = transforms.Compose([
                transforms.Resize(resolution[0], antialias=True),
                transforms.CenterCrop(resolution[0]),
                ])
        else:
            if ori_resolution is not None:
                # resize while keeping original aspect ratio,
                # then centercrop to target resolution
                resize_ratio = max(resolution[0] / ori_resolution[0], resolution[1] / ori_resolution[1])
                resolution_after_resize = [int(ori_resolution[0] * resize_ratio), int(ori_resolution[1] * resize_ratio)]
                transformations = transforms.Compose([
                    transforms.Resize(resolution_after_resize, antialias=True),
                    transforms.CenterCrop(resolution),
                    ])
            else:
                # directly resize to target resolution
                transformations = transforms.Compose([
                    transforms.Resize(resolution, antialias=True),
                    ])
    else:
        raise NotImplementedError
    return transformations

class RealEstate10K(Dataset):
    """
    RealEstate10K Dataset.
    For each video, its meta info is stored in a txt file whose contents are as follows:
    line 0: video_url
    line 1: empty
    line 2: caption

    In the rest, each line is a frame, including frame path, 4 camera intrinsics, and 3*4 camera pose (the matrix is row-major order).

    e.g.
    line 3: 0_frame_path focal_length_x focal_length_y principal_point_x principal_point_y 3*4_extrinsic_matrix 
    line 4: 1_frame_path focal_length_x focal_length_y principal_point_x principal_point_y 3*4_extrinsic_matrix
    ...
    
    meta_path: path to the meta file
    meat_list: path to the meta list file
    data_dir: path to the data folder
    video_length: length of the video clip for training
    resolution: target resolution, a list of int, [h, w]
    frame_stride: stride between frames, int or list of int, [min, max], do not larger than 32 when video_length=16
    spatial_transform: spatial transformation, ["random_crop", "resize_center_crop"]
    count_globalsteps: whether to count global steps
    bs_per_gpu: batch size per gpu, used to count global steps

    """
    def __init__(self,
                 meta_path,
                 meta_list,
                 data_dir,
                 video_length=16,
                 resolution=[256, 256],
                 frame_stride=1, # [min, max], do not larger than 32 when video_length=16
                 invert_video=True,
                 spatial_transform=None,
                 count_globalsteps=False,
                 bs_per_gpu=None,
                 RT_norm=False
                 ):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.frame_stride = frame_stride
        self.spatial_transform_type = spatial_transform
        self.count_globalsteps = count_globalsteps
        self.bs_per_gpu = bs_per_gpu
        self.invert_video = invert_video
        self.RT_norm = RT_norm

        with open(meta_list, 'r') as f:
            self.metadata = [line.strip() for line in f.readlines()]
        
        # make saptial transformations
        if isinstance(self.resolution[0], int):
            self.num_resolutions = 1
            self.spatial_transform = make_spatial_transformations(self.resolution, type=self.spatial_transform_type) \
                if self.spatial_transform_type is not None else None
        else:
            # multiple resolutions training
            assert(isinstance(self.resolution[0], list) or isinstance(self.resolution[0], omegaconf.listconfig.ListConfig))
            self.num_resolutions = len(resolution)
            self.spatial_transform = None
            self.load_raw_resolution = True
            if self.num_resolutions > 1:
                assert(self.count_globalsteps)
        if self.count_globalsteps:
            assert(bs_per_gpu is not None)
            self.counter = 0

        print(f'============= length of dataset {len(self.metadata)} =============')
        
    
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])
    
    def to_relative_RT(self, org_pose):
        org_pose = org_pose.reshape(-1, 3, 4) # [t, 3, 4]
        R_dst = org_pose[:, :, :3] # [t, 3, 3]
        T_dst = org_pose[:, :, 3:]

        R_src = np.concatenate([R_dst[0:1], R_dst[:-1]], axis=0) # [t, 3, 3]
        T_src = np.concatenate([T_dst[0:1], T_dst[:-1]], axis=0)

        R_src_inv = R_src.transpose(0, 2, 1) # [t, 3, 3]

        R_rel = R_dst @ R_src_inv # [t, 3, 3]
        T_rel = T_dst - R_rel@T_src

        RT_rel = np.concatenate([R_rel, T_rel], axis=-1) # [t, 3, 4]
        RT_rel = RT_rel.reshape(-1, 12) # [t, 12]

        return RT_rel
        
    
    def __getitem__(self, index):
        ## set up for dynamic resolution training
        if self.count_globalsteps:
            self.counter += 1
            self.global_step = self.counter // self.bs_per_gpu
        else:
            self.global_step = None

        to_inverse = (self.invert_video and random.random() > 0.5)

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            # sample = self.metadata[index]
            with open(f'{self.meta_path}/{self.metadata[index]}', 'r') as f:
                lines = f.readlines()
            caption = lines[2].strip()

            lines = lines[3:]

            frame_num = len(lines)

            if isinstance(self.frame_stride, int):
                frame_stride = self.frame_stride
            elif (isinstance(self.frame_stride, list) or isinstance(self.frame_stride, omegaconf.listconfig.ListConfig)) and len(self.frame_stride) == 2: # [min, max]
                assert(self.frame_stride[0] <= self.frame_stride[1]), f"frame_stride[0]({self.frame_stride[0]}) > frame_stride[1]({self.frame_stride[1]})"
                frame_stride = random.randint(self.frame_stride[0], self.frame_stride[1])
            else:
                print(type(self.frame_stride))
                print(len(self.frame_stride))
                print(f"frame_stride={self.frame_stride}")
                raise NotImplementedError

            required_frame_num = frame_stride * (self.video_length-1) + 1
            if frame_num < required_frame_num:
                if isinstance(self.frame_stride, int) and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length-1) + 1

            ## select a random clip
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            frames = []
            camera_pose = []
            
            for i in frame_indices:
                line = lines[i].strip().split(' ')
                frame_path = os.path.join(self.data_dir, line[0]+'.png')
                try:
                    frame = Image.open(frame_path)
                    frame = np.array(frame)
                except:
                    frames = None
                    break
                frames.append(frame)
                # camera_poses.append([float(x) for x in line[1:]])
                cur_p = np.array([float(x) for x in line[7:]]) # 3*4 extrinsic matrix
                camera_pose.append(cur_p)
                

            try:
                frames = np.stack(frames)

                if to_inverse:
                    camera_pose = camera_pose[::-1]
                if self.RT_norm:
                    camera_pose = create_relative(camera_pose, dataset="realestate")
                camera_pose = np.stack(camera_pose) # [t, 3*4]
                camera_pose = torch.tensor(camera_pose).float() # [t, 3*4]
                
            except:
                index += 1
                continue

            break

        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        
        if self.num_resolutions > 1:
            ## make transformations based on the current resolution
            res_idx = self.global_step % 3
            res_curr = self.resolution[res_idx]
            self.spatial_transform = make_spatial_transformations(res_curr, 
                                                                  self.spatial_transform_type,
                                                                  ori_resolution=frames.shape[2:])
        
        ## spatial transformations
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        if self.resolution is not None:
            if self.num_resolutions > 1:
                assert(frames.shape[2] == res_curr[0] and frames.shape[3] == res_curr[1]), f'frames={frames.shape}, res_curr={res_curr}'
            else:
                assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2

        if to_inverse:
            # inverse frame order in dim=1
            frames = frames.flip(dims=(1,))
            # camera_pose = camera_pose.flip(dims=(0,))

        data = {'video': frames, 
                'caption': caption, 
                'path': self.metadata[index], 
                'frame_stride': frame_stride, 
                'RT': camera_pose,
                'trajs': torch.zeros(2, self.video_length, frames.shape[2], frames.shape[3])}
        return data
    
    def __len__(self):
        return len(self.metadata)
