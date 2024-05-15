
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

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from motionctrl.data.utils import bivariate_Gaussian, create_relative


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
                 trajectory_type='weighted_spatially', # 'longest', 'spatially', or 'weighted_spatially'
                 expected_min_traj_len=4,
                 patch_size=16,
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
    
    def get_longest_n_trajectories(self, trajectory_label, start_idx, frame_indices, frame_stride=1):
        h, w = self.resolution
        optical_flow = np.zeros((self.video_length, h, w, 2))

        selected_traj_num = 0
        selected_traj_id = {}
        min_len = 33

        total_traj_num = random.randint(1, self.trajectory_max_num)
        for traj_id, traj in trajectory_label.items():
            if traj['labels'].all():
                cur_len = traj['frame_ids'].shape[0]
                if selected_traj_num < total_traj_num:
                    if cur_len not in selected_traj_id:
                        selected_traj_id[cur_len] = [traj_id]
                    else:
                        selected_traj_id[cur_len].append(traj_id)
                    selected_traj_num += 1
                    min_len = min(min_len, cur_len)
                else:
                    if cur_len > min_len:
                        if cur_len not in selected_traj_id:
                            selected_traj_id[cur_len] = [traj_id]
                        else:
                            selected_traj_id[cur_len].append(traj_id)
                        selected_traj_id[min_len].pop()
                        if len(selected_traj_id[min_len]) == 0:
                            del selected_traj_id[min_len]
                            min_len = min(selected_traj_id.keys())

        for k, v in selected_traj_id.items():
            for traj_id in v:
                traj = trajectory_label[traj_id]
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
                            x0, y0 = locations[i]
                            if frame_ids[i+frame_stride] in frame_indices:
                                x1, y1 = locations[i+frame_stride]
                                y0_idx = int(y0) if int(y0) < h else h-1
                                x0_idx = int(x0) if int(x0) < w else w-1
                                # y0_idx = int(y0-1)
                                # x0_idx = int(x0-1)
                                optical_flow[(frame_ids[i+frame_stride]-start_idx) // frame_stride][y0_idx, x0_idx] = np.array([x1-x0, y1-y0])
                            else:
                                continue
        return optical_flow
    
    def get_spatially_n_trajectories(self, trajectory_label, start_idx, frame_indices, frame_stride=1):
        h, w  = self.resolution
        ph, pw = h // self.patch_size, w // self.patch_size
        min_traj_len = self.expected_min_traj_len * frame_stride - 1

        patchwise_trajectory = [[{'traj': None, 'len': 0} for i in range(pw)] for j in range(ph)]

        for traj_id, traj in trajectory_label.items():
            if traj['labels'].all():
                cur_len = traj['frame_ids'].shape[0]
                if cur_len >= min_traj_len:
                    x0, y0 = traj['locations'][0]
                    # x0, y0 = int(x0-1), int(y0-1)
                    patch_h, patch_w = int(y0//self.patch_size), int(x0//self.patch_size)
                    patch_h = patch_h if patch_h < ph else ph-1
                    patch_w = patch_w if patch_w < pw else pw-1
                    patch_idx = (patch_h, patch_w)
                    if cur_len > patchwise_trajectory[patch_idx[0]][patch_idx[0]]['len']:
                        patchwise_trajectory[patch_idx[0]][patch_idx[0]]['traj'] = traj_id
                        patchwise_trajectory[patch_idx[0]][patch_idx[0]]['len'] = cur_len

        selected_trajs = []
        for i in range(ph):
            for j in range(pw):
                traj_id = patchwise_trajectory[i][j]['traj']
                if traj_id is not None:
                    selected_trajs.append(traj_id)

        optical_flow = np.zeros((self.video_length, h, w, 2))
        
        if len(selected_trajs) > 0:
            succeed = True
            # print(len(selected_trajs))
            selected_num = random.randint(1, min(len(selected_trajs), self.trajectory_max_num))
            selected_trajs = random.choices(selected_trajs, k=selected_num)

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
                            x0, y0 = locations[i]
                            if frame_ids[i+frame_stride] in frame_indices:
                                x1, y1 = locations[i+frame_stride]
                                y0_idx = int(y0) if int(y0) < h else h-1
                                x0_idx = int(x0) if int(x0) < w else w-1
                                # y0_idx = int(y0-1)
                                # x0_idx = int(x0-1)
                                optical_flow[(frame_ids[i+frame_stride]-start_idx) // frame_stride][y0_idx, x0_idx] = np.array([x1-x0, y1-y0])
                            else:
                                continue
        else:
            succeed = False
        return optical_flow, succeed
    
    def get_weighted_spatially_n_trajectories(self, trajectory_label, start_idx, frame_indices, frame_stride=1):
        h, w  = self.resolution
        ph, pw = h // self.patch_size, w // self.patch_size
        min_traj_len = self.expected_min_traj_len * frame_stride - 1

        patchwise_trajectory = [[{'traj': None, 'len': 0, 'num': 0} for i in range(pw)] for j in range(ph)]

        for traj_id, traj in trajectory_label.items():
            if traj['labels'].all():
                cur_len = traj['frame_ids'].shape[0]
                if cur_len >= min_traj_len:
                    x0, y0 = traj['locations'][0]
                    # x0, y0 = int(x0-1), int(y0-1)
                    patch_h, patch_w = int(y0//self.patch_size), int(x0//self.patch_size)
                    patch_h = patch_h if patch_h < ph else ph-1
                    patch_w = patch_w if patch_w < pw else pw-1
                    patch_idx = (patch_h, patch_w)
                    if cur_len > patchwise_trajectory[patch_idx[0]][patch_idx[0]]['len']:
                        patchwise_trajectory[patch_idx[0]][patch_idx[0]]['traj'] = traj_id
                        patchwise_trajectory[patch_idx[0]][patch_idx[0]]['len'] = cur_len
                    patchwise_trajectory[patch_idx[0]][patch_idx[0]]['num'] += 1

        selected_trajs = []
        selected_trajs_num = []
        for i in range(ph):
            for j in range(pw):
                traj_id = patchwise_trajectory[i][j]['traj']
                if traj_id is not None:
                    selected_trajs.append(traj_id)
                    selected_trajs_num.append(patchwise_trajectory[i][j]['num'])

        optical_flow = np.zeros((self.video_length, h, w, 2))
        
        if len(selected_trajs) > 0:
            succeed = True
            # print(len(selected_trajs))
            selected_num = random.randint(1, min(len(selected_trajs), self.trajectory_max_num))
            selected_idx = torch.multinomial(torch.tensor(selected_trajs_num).float(), selected_num)
            selected_trajs = [selected_trajs[i] for i in selected_idx]

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
                            x0, y0 = locations[i]
                            if frame_ids[i+frame_stride] in frame_indices:
                                x1, y1 = locations[i+frame_stride]
                                y0_idx = int(y0) if int(y0) < h else h-1
                                x0_idx = int(x0) if int(x0) < w else w-1
                                # y0_idx = int(y0-1)
                                # x0_idx = int(x0-1)
                                optical_flow[(frame_ids[i+frame_stride]-start_idx) // frame_stride][y0_idx, x0_idx] = np.array([x1-x0, y1-y0])
                            else:
                                continue
        else:
            succeed = False
        return optical_flow, succeed
    
    def get_dense_trajectories(self, trajectory_label, start_idx, frame_indices, frame_stride=1):
        h, w  = 256, 256
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
                        x0, y0 = locations[i]
                        if frame_ids[i+frame_stride] in frame_indices:
                            x1, y1 = locations[i+frame_stride]
                            y0_idx = int(y0) if int(y0) < h else h-1
                            x0_idx = int(x0) if int(x0) < w else w-1
                            traj_uv[(frame_ids[i+frame_stride]-start_idx) // frame_stride][y0_idx, x0_idx] = np.array([x1-x0, y1-y0])
                        else:
                            continue
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
            # optical flow
            # select out the longest trajectories with labels
            if self.trajectory_type == 'longest':
                optical_flow = self.get_longest_n_trajectories(trajectory_label, start_idx, frame_indices, frame_stride=frame_stride)
                succeed = True
            elif self.trajectory_type == 'spatially':
                optical_flow, succeed = self.get_spatially_n_trajectories(trajectory_label, start_idx, frame_indices, frame_stride=frame_stride)
            elif self.trajectory_type == 'weighted_spatially':
                optical_flow, succeed = self.get_weighted_spatially_n_trajectories(trajectory_label, start_idx, frame_indices, frame_stride=frame_stride)
            elif self.trajectory_type == 'dense':
                optical_flow, succeed = self.get_dense_trajectories(trajectory_label, start_idx, frame_indices, frame_stride=frame_stride)
            else:
                raise NotImplementedError

            if not succeed:
                print(f'No trajectory found! path = {sample_dir}')
                index = (index + 1) % len(self.metadata)
                continue
            
            if self.trajectory_type != 'dense':
                for i in range(1, self.video_length):
                    optical_flow[i] = cv2.filter2D(optical_flow[i], -1, self.blur_kernel)
                    
            optical_flow = torch.tensor(optical_flow).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

            # ------------------------------------------------------------------
            # caption
            parts = sample_dir.split('/')
            page_dir = parts[-2]
            videoid = int(parts[-1].split('_')[0])
            caption = self.metadata_org.loc[(page_dir, videoid)]['caption']

            # ------------------------------------------------------------------
            # video
            frames = []
            for frame_idx in frame_indices:
                try:
                    frame = Image.open(f'{sample_dir}/images/{frame_idx:05d}.png')
                except:
                    print(f'Load image failed! path = {sample_dir}')
                    index = (index + 1) % len(self.metadata)
                    continue
                frame = np.array(frame)
                frames.append(frame)
            frames = np.stack(frames)

            ## process data
            assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
            frames = torch.tensor(frames).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
            frames = (frames / 255 - 0.5) * 2

            # ------------------------------------------------------------------
            # camera pose
            camera_pose = []
            for frame_idx in frame_indices:
                try:
                    camera_pose.append(np.loadtxt(f'{sample_dir}/colmap_outputs_converted/poses/{frame_idx:05d}.txt'))
                except:
                    print(f'Load camera pose failed! path = {sample_dir}')
                    index = (index + 1) % len(self.metadata)
                    continue
            camera_pose = create_relative(camera_pose, dataset="syn", K_1=470/1.5) #scale_T=0.2
            camera_pose = np.stack(camera_pose, axis=0) # [t,3,4]

            camera_pose = torch.tensor(camera_pose).float() # [t,3,4]
            camera_pose = camera_pose.reshape(self.video_length, -1) # [t,12]

            # if torch.isinf(camera_pose).any() or torch.isneginf(camera_pose).any():
            #     print(f'inf in camera pose! path = {sample_dir}')
            #     print(camera_pose)
            #     index = (index + 1) % len(self.metadata)
            #     continue

            

            break

        data = {'video': frames, 
                'caption': caption, 
                'path': sample_dir, 
                'frame_stride': frame_stride, 
                'trajs': optical_flow,
                'RT': camera_pose}
        return data
    
    def __len__(self):
        return len(self.metadata)

if __name__== "__main__":
    # dataset from config
    # data_config= "myscripts/train_text2video/tv_032_ImgJointTrainDynFPS_2DAE_basedon024/tv_032_ImgJointTrainDynFPS_2DAE_basedon024.yaml"
    # configs = OmegaConf.load(data_config)
    # dataset = instantiate_from_config(configs['data']['params']['train'])
    
    meta_path='/group/30042/public_datasets/WebVid/meta_data/results_2M_train.csv'
    data_dir='/group/30042/public_datasets/WebVid'
    motion_seg_list='/group/30042/public_datasets/WebVid/motion_seg_list_v1.txt'
    dataset = WebVid(meta_path,
                 data_dir,
                 motion_seg_list=motion_seg_list,
                 subsample=None,
                 video_length=16,
                 resolution=[256,256],
                 frame_stride=[1,2],
                 spatial_transform=None,
                 fps_max=None,
                 load_raw_resolution=True,
                 trajectory_type='weighted_spatially',
                 )
    # dataset.getitem(0)
    
    dataloader = DataLoader(dataset,
                    batch_size=8,
                    num_workers=12,
                    shuffle=False)
    # print(f'dataset len={dataset.__len__()}')
    # i=0
    # total_fps = set()
    # res1 = [336,596]
    # res2 = [316, 600]
    # n_videos_res1=0
    # n_videos_res2=0
    # n_videos_misc=0
    # other_res = []

    from tqdm import tqdm
    for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):
        pass
        # pass
        # print(f"video={batch['video'].shape}, fps={batch['fps']}")
        # # total_fps.add(batch['fps'].item())
        # if batch['video'].shape[-2] == res1[0] and batch['video'].shape[-1] == res1[1]:
        #     n_videos_res1 += 1
        # elif batch['video'].shape[-2] == res2[0] and batch['video'].shape[-1] == res2[1]:
        #     n_videos_res2 += 1
        # else:
        #     n_videos_misc += 1
        #     other_res.append(list(batch['video'].shape[3:]))

        # if (i + 1) == 1000:
        #     break

    # print(f'total videos = {i}')
    # print('======== total_fps ========')
    # print(total_fps)
    # print('======== resolution ========')
    # print(f'res1 {res1}: n_videos = {n_videos_res1}')
    # print(f'res2 {res2}: n_videos = {n_videos_res2}')
    # print(f'other resolution: n_videos = {n_videos_misc}')
    # print(f'other resolutions: {other_res}')

