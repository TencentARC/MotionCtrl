import json
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from motionctrl.data.utils import create_relative


class Objaverse360(Dataset):
    def __init__(
        self,
        data_dir='.objaverse/hf-objaverse-v1/views',
        info_file='UniG3D-Objaverse_caption_CLIP_25.0.json',
        video_length=16,
        resolution=[256, 256],
        frame_stride=1, # equal to half_rotate, frame_stride=2 means 360 degree
        invert_video=True,
        **kwargs
    ):
        self.data_dir = data_dir
        self.video_length = video_length
        self.resolution = resolution
        self.frame_stride = frame_stride
        self.frame_num = 32
        self.invert_video = invert_video

        self.transform = transforms.Compose([
            transforms.Resize(max(resolution), antialias=True),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        ])

        with open(os.path.join(data_dir, info_file)) as f:
            self.captions = json.load(f)

        self.metadata = list(self.captions.keys())

        print('============= length of dataset %d =============' % len(self.metadata))

    def load_im(self, path, color=[1., 1., 1., 1.]):
        '''
        replace background pixel with random color in rendering
        '''
        img = plt.imread(path)
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        img = img.convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        while True:
            uid = self.metadata[index]
            video_path = os.path.join(self.data_dir, 'views_circular', uid)
            caption = self.captions[uid]['caption']
            caption = caption + ', rotating around its axis, 360 degrees. White background. 3D render.'

            to_inverse = (self.invert_video and random.random() > 0.5)

            color = [1., 1., 1., 1.]
            try:
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

                assert(frame_stride ==1 or frame_stride ==2), f'frame_stride={frame_stride}'
                required_frame_num = frame_stride * (self.video_length-1) + 1
                
                ## select a random clip
                random_range = self.frame_num - required_frame_num
                start_idx = random.randint(0, random_range) if random_range > 0 else 0
                frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]

                # get frames
                frames = []
                for i in frame_indices:
                    frame = self.load_im(os.path.join(video_path, '%03d.png' % i), color)
                    frames.append(frame)
                frames = torch.stack(frames, 0).permute(1, 0, 2, 3).float() # [t,c,h,w] -> [c,t,h,w]
        
                # get camera pose
                camera_pose = []
                for i in frame_indices:
                    camera_pose.append(np.load(os.path.join(video_path, '%03d.npy' % i))) # [3,4]
                
                if to_inverse:
                    frames = frames.flip(dims=(1,))
                    camera_pose = camera_pose[::-1]

                camera_pose = create_relative(camera_pose, dataset="zero123")
                camera_pose = np.stack(camera_pose, axis=0) # [t,3,4]

                camera_pose = torch.tensor(camera_pose).float() # [t,3,4]
                camera_pose = camera_pose.reshape(self.video_length, -1) # [t,12]

            except:
                index = (index + 1) % len(self.metadata)
                # index = random.randint(0, len(self.metadata) - 1)
                print(f"Load video failed! path = {video_path}")
                continue

            if torch.isinf(camera_pose).any() or torch.isneginf(camera_pose).any():
                print(f'inf in camera pose! path = {video_path}')
                print(f'camera_pose={camera_pose}')
                index = (index + 1) % len(self.metadata)
                continue
            
            if frames.shape[1] == self.video_length:
                break


        data = {}
        data['video'] = frames
        data['caption'] = caption
        data['path'] = video_path
        # data['fps'] = 16
        data['frame_stride'] = frame_stride
        data['RT'] = camera_pose
        data['trajs'] = torch.zeros(2, self.video_length, frames.shape[2], frames.shape[3])
        return data


if __name__ == '__main__':
    root_out_dir = '/group/30098/zhouxiawang/outputs/LDVMPose/tmp2'
    os.makedirs(root_out_dir, exist_ok=True)

    video_length = 14
    resolution = [576, 1024]

    dataset = Objaverse360(
        data_dir='/group/30042/public_datasets/3d_datasets/objaverse/views_release',
        video_length=video_length, 
        resolution=resolution
    )
    dataloader = DataLoader(
        dataset,
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
        out_dir = f'{root_out_dir}/{batch["path"][i]}'
        os.makedirs(out_dir, exist_ok=True)

        frames = batch['video'][i].permute(1,2,3,0).numpy()
        frames = (frames / 2 + 0.5) * 255
        frames = frames.astype(np.uint8)
        for i in range(frames.shape[0]):
            Image.fromarray(frames[i]).save(f'{out_dir}/{i:5d}.png')