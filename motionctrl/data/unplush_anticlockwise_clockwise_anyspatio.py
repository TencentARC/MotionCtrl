import math
import os
import random
import sys
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from motionctrl.data.utils import create_relative


class UnplushDataset(Dataset):
    def __init__(self,
                 data_dir='/group/30042/public_datasets/unsplash_resize',
                 video_length=16,
                 resolution=[256, 256],
                 invert_video=True,
                 max_degree=120,
                 min_stride=1,

                 pose_type='orginal', # 'orginal' or 'relative_RT2'
                 ):
        self.data_dir = data_dir
        self.max_degree = max_degree
        self.pose_type = pose_type
        self.min_stride = min_stride

        # self.metadata = glob(f'{self.data_dir}/unsplash-from-1*-to-1*_caption/*txt')
        # self.metadata = glob(f'{self.data_dir}/unsplash-from-0-to-100_caption/*txt')
        self.metadata = glob(f'{self.data_dir}/unsplash-from-1000000-to-1050000_caption/*txt') # 50000

        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.max_resolution = int(math.sqrt(self.resolution[0] * self.resolution[0] + 
                                        self.resolution[1] * self.resolution[1]))
        self.transform_step1 = transforms.Compose([
                # transforms.Resize(self.max_resolution, antialias=True), # sqrt(128^2 + 128^2) * 2
                transforms.CenterCrop(self.max_resolution),
                ])
        self.transform_step2 = transforms.Compose([
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor()
                ])
        self.max_stride = max_degree // self.video_length
        assert self.max_stride >= self.min_stride, f'max_stride={self.max_stride}, min_stride={self.min_stride}'

        self.invert_video = invert_video
        print(f'============= length of dataset {len(self.metadata)} =============')

    def compute_R_form_rad_angle(self, angles):
        theta_x, theta_y, theta_z = angles
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
        
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
        
        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])
        
        # 计算相机外参的旋转矩阵
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R
    
    def __getitem__(self, index):

        while True:
            try:
                caption_path = self.metadata[index]
            except:
                index = (index + 1) % len(self.metadata)
                print(f'fail to load {caption_path}')
                continue
            with open(caption_path, 'r') as f:
                lines = f.readlines()
            caption = lines[0].strip()

            jpg_path = caption_path.replace('_caption/', '/').replace('.txt', '.jpg')
            img = Image.open(jpg_path)

            w, h = img.size
            h_ratio = h * 1.0 / self.max_resolution
            w_ratio = w * 1.0 / self.max_resolution

            if h_ratio > w_ratio:
                h = int(h / w_ratio)
                if h < self.max_resolution:
                    h = self.max_resolution
                w = self.max_resolution
            else:
                w = int(w / h_ratio)
                if w < self.max_resolution:
                    w = self.max_resolution
                h = self.max_resolution
            img = transforms.Resize((h, w))(img)

            img = self.transform_step1(img)

            angle_stride = random.randint(self.min_stride, self.max_stride)
            angle_degs = [angle_stride * i for i in range(self.video_length)]
            if self.invert_video and random.random() > 0.5:
                angle_degs = [-x for x in angle_degs]
            
            frames = []
            RTs = []
            for angle_deg in angle_degs:
                img_rotate = img.rotate(angle_deg)
                img_crop = self.transform_step2(img_rotate) # [c,h,w] [0,1]
                frames.append(img_crop)

                angle_rad = 2*np.pi*(angle_deg/360)
                R = self.compute_R_form_rad_angle([0, 0, -1*angle_rad]) 
                T = np.array([0, 0, 0]).reshape(3,1)
                RT = np.concatenate([R,T], axis=1).reshape(-1)
                RTs.append(RT)

            ## process data
            frames = torch.stack(frames, dim=0) # [t,c,h,w]
            assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
            frames = frames.permute(1,0,2,3).float() # [t,c,h,w] -> [c,t,h,w]
            frames = (frames - 0.5) * 2 # [-1,1]

            ## process pose
            RTs = np.stack(RTs, axis=0) # [t,12]
            RTs = torch.from_numpy(RTs).float() # [t,12]

            break

        data = {
            'video': frames, 
            'caption': caption, 
            'path': self.metadata[index], 
            'frame_stride': 1, 
            'RT': RTs,
            'trajs': torch.zeros(2, self.video_length, frames.shape[2], frames.shape[3])}
        
        return data
    
    def __len__(self):
        return len(self.metadata)

if __name__== "__main__":
    # dataset from config
    # data_config= "myscripts/train_text2video/tv_032_ImgJointTrainDynFPS_2DAE_basedon024/tv_032_ImgJointTrainDynFPS_2DAE_basedon024.yaml"
    # configs = OmegaConf.load(data_config)
    # dataset = instantiate_from_config(configs['data']['params']['train'])
    
    root_out_dir = '/group/30098/zhouxiawang/outputs/LDVMPose/tmpacw2'
    os.makedirs(root_out_dir, exist_ok=True)
    meta_path="/group/30098/public_datasets/3d/RealEstate10K/caption/test"
    meta_list='/group/30098/public_datasets/3d/RealEstate10K/caption/test.txt'
    data_dir="/group/30098/public_datasets/3d/RealEstate10K/process/test"
    dataset = UnplushDataset(
        video_length=14,
        resolution=[576, 1024],
        invert_video=True,
        max_degree=120,
    )
    dataloader = DataLoader(dataset,
                    batch_size=4,
                    num_workers=0,
                    shuffle=False)
    
    batch = next(iter(dataloader))
    print(batch['video'].min(), batch['video'].max())
    print(batch['caption'])
    print(batch['path'])
    print(batch['fps_id'])
    # print(batch['frame_stride'])

    for i in range(4):
        out_dir = f'{root_out_dir}/{batch["path"][i].split("/")[-1]}'
        os.makedirs(out_dir, exist_ok=True)

        frames = batch['video'][i].permute(1,2,3,0).numpy()
        frames = (frames / 2 + 0.5) * 255
        frames = frames.astype(np.uint8)
        for i in range(frames.shape[0]):
            Image.fromarray(frames[i]).save(f'{out_dir}/{i:5d}.png')

    # print(f'dataset len={dataset.__len__()}')
    # i=0

    # # for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):
    # for i in range(4, 8, 1):
    #     # pass
    #     import pdb
    #     pdb.set_trace()
    #     batch = dataset.getitem(i)
    #     print(f"video={batch['video'].shape}")
    #     print(f"caption={batch['caption']}")
    #     print(f"path={batch['path']}")
    #     # print(f"frame_stride={batch['frame_stride']}")
    #     print(f"T={batch['RT'].shape}")
    #     print(f'T={batch["RT"]}')
        
    #     out_dir = f'{root_out_dir}/{batch["path"][0].split("/")[-1]}'
    #     os.makedirs(out_dir, exist_ok=True)

    #     # frames = batch['video'][0].permute(1,2,3,0).numpy()
    #     frames = batch['video'].permute(1,2,3,0).numpy()
    #     frames = (frames / 2 + 0.5) * 255
    #     frames = frames.astype(np.uint8)
    #     for i in range(frames.shape[0]):
    #         Image.fromarray(frames[i]).save(f'{out_dir}/{i:5d}.png')

    #     # break