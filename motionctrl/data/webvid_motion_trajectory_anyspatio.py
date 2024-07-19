
import os
import random
import sys

import cv2
import numpy as np
import omegaconf
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from decord import VideoReader, cpu

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from motionctrl.data.utils import bivariate_Gaussian


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
                transforms.Resize(resolution[0]),
                transforms.CenterCrop(resolution[0]),
                ])
        else:
            if ori_resolution is not None:
                # resize while keeping original aspect ratio,
                # then centercrop to target resolution
                resize_ratio = max(resolution[0] / ori_resolution[0], resolution[1] / ori_resolution[1])
                resolution_after_resize = [int(ori_resolution[0] * resize_ratio), int(ori_resolution[1] * resize_ratio)]
                transformations = transforms.Compose([
                    transforms.Resize(resolution_after_resize),
                    transforms.CenterCrop(resolution),
                    ])
            else:
                # directly resize to target resolution
                transformations = transforms.Compose([
                    transforms.Resize(resolution),
                    ])
    elif type == "align2_256":
        is_square = (resolution[0] == resolution[1])
        if is_square:
            transformations = transforms.Compose([
                transforms.Resize(resolution[0]),
                transforms.CenterCrop(resolution[0]),
                ])
        else:
            transformations = transforms.Compose([
                transforms.Resize(max(resolution)),
                transforms.CenterCrop(resolution),
                ])
    else:
        raise NotImplementedError
    return transformations

class WebVid(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 motion_seg_list=None,
                 trajectory_max_num=8,
                 blur_size=99,
                 blur_sigma=10,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 256],
                 frame_stride=1,
                 spatial_transform=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 count_globalsteps=False,
                 bs_per_gpu=None,
                 trajectory_type='longest', # 'longest', 'spatially', or 'weighted_spatially'
                 expected_min_traj_len=4,
                 patch_size=16,
                 zero_poses=False,
                 **kwargs
                 ):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.motion_seg_list = motion_seg_list
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.spatial_transform_type = spatial_transform
        self.count_globalsteps = count_globalsteps
        self.bs_per_gpu = bs_per_gpu
        self.frame_num = 32
        self.trajectory_type = trajectory_type
        self.expected_min_traj_len = expected_min_traj_len
        self.patch_size = patch_size
        self.zero_poses = zero_poses

        self.trajectory_max_num = trajectory_max_num
        # self.blur_size = blur_size

        self.blur_kernel = bivariate_Gaussian(blur_size, blur_sigma, blur_sigma, 0, grid=None, isotropic=True)

        self._load_metadata()

        print('*'*80)
        print(f'WebVid dataset: {len(self.metadata)} videos')
        print('*'*80)

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

    def _load_metadata(self):
        with open(self.motion_seg_list, 'r') as f:
            motion_seg_list = f.readlines()
        self.metadata = [x.strip() for x in motion_seg_list]
        
        # str: page_dir
        # int: videoid
        metadata = pd.read_csv(self.meta_path, index_col=['page_dir', 'videoid'])
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata_org = metadata
        self.metadata_org.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]
    
    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        # full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        full_video_fp = os.path.join(self.data_dir, 'train', rel_video_fp)
        return full_video_fp, rel_video_fp
   
    def get_sparse_trajectories(self, trajectory_label, start_idx, 
                                frame_indices, frame_stride=1,
                                resolution=[256, 256]):
        org_h, org_w  = 256, 256
        target_h, target_w = resolution
        if resolution == [org_h, org_w]:
            h, w = org_h, org_w
            scale = 1
            del_len = 0
        else:
            # only support target_h <= target_w
            h = w = target_w
            scale = h / org_h
            del_len = int((h - target_h) / 2) # [0, del_len] and [-del_len:] will be deleted
        
        # traj_uv = np.zeros((self.video_length, h, w, 2))

        ph, pw = h // self.patch_size, w // self.patch_size
        min_traj_len = self.expected_min_traj_len * frame_stride - 1

        patchwise_trajectory = [[{'traj': [], 'len': [], 'num': 0} for i in range(pw)] for j in range(ph)]

        for traj_id, traj in trajectory_label.items():
            if traj['labels'].all():
                cur_len = traj['frame_ids'].shape[0]
                if cur_len >= min_traj_len:
                    x0, y0 = traj['locations'][0] * scale
                    if y0 > del_len and y0 < h-del_len:
                        # x0, y0 = int(x0-1), int(y0-1)
                        patch_h, patch_w = int(y0//self.patch_size), int(x0//self.patch_size)
                        patch_h = patch_h if patch_h < ph else ph-1
                        patch_w = patch_w if patch_w < pw else pw-1
                        patch_idx = (patch_h, patch_w)
                        # if cur_len > patchwise_trajectory[patch_idx[0]][patch_idx[0]]['len']:
                        patchwise_trajectory[patch_idx[0]][patch_idx[0]]['traj'].append(traj_id)
                        patchwise_trajectory[patch_idx[0]][patch_idx[0]]['len'].append(cur_len)
                        patchwise_trajectory[patch_idx[0]][patch_idx[0]]['num'] += 1

        selected_trajs = []
        selected_trajs_num = []
        for i in range(ph):
            for j in range(pw):
                traj_id = patchwise_trajectory[i][j]['traj']
                if traj_id is not None:
                    selected_trajs.append(traj_id)
                    selected_trajs_num.append(patchwise_trajectory[i][j]['num'])

        traj_uv = np.zeros((self.video_length, h, w, 2))
        
        if len(selected_trajs) > 0:
            succeed = True
            print('------------------')
            print(len(selected_trajs))
            selected_num = random.randint(1, min(len(selected_trajs), self.trajectory_max_num))
            print(selected_num)
            print(selected_trajs_num)
            selected_idx = torch.multinomial(torch.tensor(selected_trajs_num).float(), selected_num)
            print(selected_idx)
            selected_trajs = [selected_trajs[i] for i in selected_idx]
            selected_trajs = [s[random.randint(0, len(s)-1)] for s in selected_trajs]
            # print(selected_trajs)

            for traj_id in selected_trajs:
                traj = trajectory_label[traj_id]
                locations = traj['locations']
                frame_ids = traj['frame_ids']
                sidx = 0
                while True:
                    if frame_ids[sidx] in frame_indices:
                        break
                    if sidx == len(frame_ids)-1:
                        break
                    sidx += 1

                if (len(frame_ids) - sidx - 1) // frame_stride >= 1:
                    for i in range(sidx, len(frame_ids)-frame_stride-1, frame_stride):
                        x0, y0 = locations[i] * scale
                        if frame_ids[i+frame_stride] in frame_indices:
                            x1, y1 = locations[i+frame_stride] * scale
                            y0_idx = int(y0) if int(y0) < h else h-1
                            x0_idx = int(x0) if int(x0) < w else w-1
                            # y0_idx = int(y0-1)
                            # x0_idx = int(x0-1)
                            traj_uv[(frame_ids[i+frame_stride]-start_idx) // frame_stride][y0_idx, x0_idx] = np.array([x1-x0, y1-y0])
                        else:
                            continue

            traj_uv = traj_uv[:, del_len:h-del_len, :, :]
        else:
            succeed = False
        return traj_uv, succeed
    
    def get_sparse_trajectories_v2(self, trajectory_label, 
                               start_idx, frame_indices, 
                               frame_stride=1,
                               resolution=[256, 256]):
        org_h, org_w  = 256, 256
        target_h, target_w = resolution
        if resolution == [org_h, org_w]:
            h, w = org_h, org_w
            scale = 1
            del_len = 0
        else:
            # only support target_h <= target_w
            h = w = target_w
            scale = h / org_h
            del_len = int((h - target_h) / 2) # [0, del_len] and [-del_len:] will be deleted
        
        traj_uv = np.zeros((self.video_length, h, w, 2))
        min_traj_len = self.expected_min_traj_len * frame_stride - 1

        # optical flow
        useful_traj_id = []
        useful_traj_speed = []
        for traj_id, traj in trajectory_label.items():
            labels = traj['labels']
            locations = traj['locations']
            frame_ids = traj['frame_ids']
            if (labels).all() and traj['frame_ids'].shape[0] >= min_traj_len:
                sidx = 0
                while True:
                    if frame_ids[sidx] in frame_indices:
                        break
                    if sidx == len(frame_ids)-1:
                        break
                    sidx += 1
                
                useful_traj_id.append(traj_id)
                useful_traj_speed.append(0)

                if (len(frame_ids) - sidx - 1) // frame_stride >= 1:
                    for i in range(sidx, len(frame_ids)-frame_stride-1, frame_stride):
                        x0, y0 = locations[i] * scale
                        if y0 > del_len and y0 < h-del_len:
                            if frame_ids[i+frame_stride] in frame_indices:
                                x1, y1 = locations[i+frame_stride] * scale
                                y0_idx = int(y0) if int(y0) < h else h-1
                                x0_idx = int(x0) if int(x0) < w else w-1
                                useful_traj_speed[-1] += np.linalg.norm(np.array([x1-x0, y1-y0]))
                                # traj_uv[(frame_ids[i+frame_stride]-start_idx) // frame_stride][y0_idx, x0_idx] = np.array([x1-x0, y1-y0])
                            else:
                                continue
                        else:
                            useful_traj_speed.pop()
                            useful_traj_id.pop()
                            break
        # import pdb; pdb.set_trace()
        if len(np.where(np.array(useful_traj_speed) > 0)[0]) == 0:
            succeed = False
            return traj_uv, succeed
        
        if len(useful_traj_id) > 0:
            succeed = True
            
            selected_num = random.randint(1, min(len(useful_traj_id), self.trajectory_max_num))
            selected_idx = torch.multinomial(torch.tensor(useful_traj_speed).float(), selected_num)
            selected_trajs = [useful_traj_id[i] for i in selected_idx]

            for traj_id in selected_trajs:
                traj = trajectory_label[traj_id]
                locations = traj['locations']
                frame_ids = traj['frame_ids']
                sidx = 0
                while True:
                    if frame_ids[sidx] in frame_indices:
                        break
                    if sidx == len(frame_ids)-1:
                        break
                    sidx += 1

                if (len(frame_ids) - sidx - 1) // frame_stride >= 1:
                    for i in range(sidx, len(frame_ids)-frame_stride-1, frame_stride):
                        x0, y0 = locations[i] * scale
                        if frame_ids[i+frame_stride] in frame_indices:
                            x1, y1 = locations[i+frame_stride] * scale
                            y0_idx = int(y0) if int(y0) < h else h-1
                            x0_idx = int(x0) if int(x0) < w else w-1
                            # y0_idx = int(y0-1)
                            # x0_idx = int(x0-1)
                            traj_uv[(frame_ids[i+frame_stride]-start_idx) // frame_stride][y0_idx, x0_idx] = np.array([x1-x0, y1-y0])
                        else:
                            continue

            traj_uv = traj_uv[:, del_len:h-del_len, :, :]
        else:
            succeed = False
        return traj_uv, succeed


    def get_dense_trajectories(self, trajectory_label, 
                               start_idx, frame_indices, 
                               frame_stride=1,
                               resolution=[256, 256]):
        org_h, org_w  = 256, 256
        target_h, target_w = resolution
        if resolution == [org_h, org_w]:
            h, w = org_h, org_w
            scale = 1
            del_len = 0
        else:
            # only support target_h <= target_w
            h = w = target_w
            scale = h / org_h
            del_len = int((h - target_h) / 2) # [0, del_len] and [-del_len:] will be deleted
        
        traj_uv = np.zeros((self.video_length, h, w, 2))

        # optical flow
        for traj_id, traj in trajectory_label.items():
            labels = traj['labels']
            locations = traj['locations']
            frame_ids = traj['frame_ids']
            if (labels).all():
                sidx = 0
                while True:
                    if frame_ids[sidx] in frame_indices:
                        break
                    if sidx == len(frame_ids)-1:
                        break
                    sidx += 1

                if (len(frame_ids) - sidx - 1) // frame_stride >= 1:
                    for i in range(sidx, len(frame_ids)-frame_stride-1, frame_stride):
                        x0, y0 = locations[i] * scale
                        if y0 > del_len and y0 < h-del_len:
                            if frame_ids[i+frame_stride] in frame_indices:
                                x1, y1 = locations[i+frame_stride] * scale
                                y0_idx = int(y0) if int(y0) < h else h-1
                                x0_idx = int(x0) if int(x0) < w else w-1
                                traj_uv[(frame_ids[i+frame_stride]-start_idx) // frame_stride][y0_idx, x0_idx] = np.array([x1-x0, y1-y0])
                            else:
                                continue

        traj_uv = traj_uv[:, del_len:h-del_len, :, :]
        
        return traj_uv, True

    def __getitem__(self, index):
        ## set up for dynamic resolution training
        if self.count_globalsteps:
            self.counter += 1
            self.global_step = self.counter // self.bs_per_gpu
        else:
            self.global_step = None

        index = index % len(self.metadata)

        while True:
            try:
                sample_dir = self.metadata[index]
                trajectory_label = np.load(f'{sample_dir}/trajectories_labeled/track.npy', allow_pickle=True).item()
            except:
                print(f'Load trajectory label failed! path = {sample_dir}')
                index = (index + 1) % len(self.metadata)
                continue

            if (isinstance(self.frame_stride, list) or isinstance(self.frame_stride, omegaconf.listconfig.ListConfig)):
                frame_stride = random.randint(self.frame_stride[0], self.frame_stride[1]) # [self.frame_stride[0], self.frame_stride[1]]
            else:
                frame_stride = self.frame_stride

            required_frame_num = frame_stride * (self.video_length-1) + 1
            
            ## select a random clip
            random_range = self.frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]

            # ------------------------------------------------------------------
            if self.trajectory_type == 'sparse':
                traj_uv, succeed = self.get_sparse_trajectories_v2(trajectory_label, 
                                                                start_idx, 
                                                                frame_indices, 
                                                                frame_stride=frame_stride,
                                                                resolution=self.resolution)
                for i in range(1, self.video_length):
                    traj_uv[i] = cv2.filter2D(traj_uv[i], -1, self.blur_kernel)
            elif self.trajectory_type == 'dense':
                traj_uv, succeed = self.get_dense_trajectories(trajectory_label, 
                                                               start_idx, 
                                                               frame_indices, 
                                                               frame_stride=frame_stride,
                                                               resolution=self.resolution)
            else:
                raise NotImplementedError

            if not succeed:
                print(f'No trajectory found! path = {sample_dir}')
                index = (index + 1) % len(self.metadata)
                continue
                    
            traj_uv = torch.tensor(traj_uv).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

            # ------------------------------------------------------------------
            # caption
            parts = sample_dir.split('/')
            page_dir = parts[-2]
            videoid, global_start_idx, global_stride = parts[-1].split('_')
            videoid = int(videoid)
            global_start_idx = int(global_start_idx)
            global_stride = int(global_stride)
            caption = self.metadata_org.loc[(page_dir, videoid)]['caption']

            # ------------------------------------------------------------------
            # video
            rel_video_fp = os.path.join(page_dir, str(videoid) + '.mp4')
            video_path = os.path.join(self.data_dir, 'train', rel_video_fp)

            try: 
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])

            except:
                print(f'Load video failed! path = {video_path}')
                index = (index + 1) % len(self.metadata)
                continue

            # fps_ori = video_reader.get_avg_fps()
            frame_stride = global_stride * frame_stride
            start_idx = global_start_idx + start_idx
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]

            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f'Load video failed! path = {sample_dir}')
                index += 1
                continue

        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        
        if self.load_raw_resolution:
            ## make transformations based on the current resolution
            self.spatial_transform = make_spatial_transformations(self.resolution, 
                                                                  self.spatial_transform_type,
                                                                  ori_resolution=frames.shape[2:])
        
        
        ## spatial transformations
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        if self.resolution is not None:
            if self.num_resolutions > 1:
                assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, res_curr={self.resolution}'
            else:
                assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2

        if self.zero_poses:
            RTs = torch.zeros((self.video_length, 12))
        else:
            RT = np.array([ 1,0,0,0,
                            0,1,0,0,
                            0,0,1,0])
            RTs = [RT] * self.video_length
            RTs = np.stack(RTs, axis=0) # [t,12]
            RTs = torch.from_numpy(RTs).float() # [t,12]

        data = {'video': frames, 
                'caption': caption, 
                'path': sample_dir, 
                'frame_stride': frame_stride, 
                'trajs': traj_uv,
                'RT': RTs}
        return data
    
    def __len__(self):
        return len(self.metadata)

