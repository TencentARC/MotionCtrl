import argparse
import glob
import math
import os
import random
import shutil
import time

import cv2
import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from torchvision import transforms
from tqdm import tqdm

import torch


def parse_args():
    parser.add_argument("--workspace_dir", type=str, default="none", help="input workspace")
    parser.add_argument("--image_folder", type=str, default="images", help="image folder") # also used in the folder option

    parser.add_argument("--start_idx", type=int, default=0, help='the starting index of the sequence')
    parser.add_argument("--end_idx", type=int, default=math.inf, help='the ending index of the sequence')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    resolution = 256
    transformations = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        ])
    
    frame_stride_range = (1, 16)
    video_length = 32
    min_video_length = 64

    dataset = 'train'
    meta_file = f'WebVid/meta_data/results_10M_{dataset}.csv'
    metadata = pd.read_csv(meta_file)
    data_dir = f'WebVid/{dataset}'

    # easy to process a subset of the dataset in different machines parallelly
    start_idx = args.start_idx
    end_idx = args.end_idx
    start_idx = 0 if start_idx < 0 else start_idx
    end_idx = len(metadata) if end_idx > len(metadata) else end_idx

    output_root = f'WebVid/{dataset}_256_32' #_{start_idx}_{end_idx}'
    os.makedirs(output_root, exist_ok=True)

    
    for idx in tqdm(range(start_idx, end_idx)):
        sample = metadata.iloc[idx]
        try:
            rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
            video_path = os.path.join(data_dir, rel_video_fp)
            video_reader = VideoReader(video_path, ctx=cpu(0))
        except:
            continue
        frame_num = len(video_reader)

        if frame_num < min_video_length:
            print(f'Video {video_path} has only {frame_num} frames, which is less than {min_video_length}.')
            continue

        frame_stride = random.randint(frame_stride_range[0], frame_stride_range[1])
        required_frame_num = (video_length-1) * frame_stride + 1
        while required_frame_num > frame_num:
            frame_stride -= 1
            required_frame_num = (video_length-1) * frame_stride + 1
        
        ## select a random clip
        random_range = frame_num - required_frame_num
        sidx = random.randint(0, random_range) if random_range > 0 else 0
        frame_indices = [sidx + frame_stride*i for i in range(video_length)]

        try:
            frames = video_reader.get_batch(frame_indices)
        except:
            print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
            continue

        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        frames = transformations(frames)
        frames = frames.permute(1, 2, 3, 0)
        frames = frames.numpy().astype(np.uint8)

        # page_dir/videoid_start-idx_frame-stride
        args.workspace_dir = os.path.join(output_root, sample['page_dir'], str(sample['videoid'])+'_'+str(sidx)+'_'+str(frame_stride))
        os.makedirs(args.workspace_dir, exist_ok=True)
        img_dir = os.path.join(args.workspace_dir, args.image_folder)
        
        # if the images already exist, skip
        if os.path.exists(img_dir):
            imgs = glob.glob(os.path.join(img_dir, '*.png'))
            if len(imgs) == video_length:
                continue

        os.makedirs(img_dir, exist_ok=True)

        for i in range(frames.shape[0]):
            img = frames[i]
            img = img[:, :, ::-1]
            cv2.imwrite(os.path.join(img_dir, f'{i:05d}.png'), img)