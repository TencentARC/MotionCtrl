import bisect
import os
import random

import omegaconf
import pandas as pd
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


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
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 spatial_transform=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 count_globalsteps=False,
                 bs_per_gpu=None,
                 ):
        self.meta_path = meta_path
        self.data_dir = data_dir
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
        self._load_metadata()
        
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
        metadata = pd.read_csv(self.meta_path)
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]
    
    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        # full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        full_video_fp = os.path.join(self.data_dir, 'train', rel_video_fp)
        return full_video_fp, rel_video_fp
    
    def __getitem__(self, index):
        ## set up for dynamic resolution training
        if self.count_globalsteps:
            self.counter += 1
            self.global_step = self.counter // self.bs_per_gpu
        else:
            self.global_step = None

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            video_path, rel_fp = self._get_video_path(sample)
            caption = sample['caption']

            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
                if len(video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(self.frame_stride * (1.0 * fps_ori / self.fixed_fps))
            else:
                frame_stride = self.frame_stride
            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:                    
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length-1) + 1
            else:
                pass

            ## select a random clip
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue
        
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        
        if self.num_resolutions > 1:
            ## make transformations based on the current resolution
            res_idx = self.global_step % 3
            res_curr = self.resolution[res_idx]
            self.spatial_transform = make_spatial_transformations(res_curr, 
                                                                  self.spatial_transform_type,
                                                                  ori_resolution=frames.shape[2:])
        else:
            pass
        ## spatial transformations
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        if self.resolution is not None:
            if self.num_resolutions > 1:
                assert(frames.shape[2] == res_curr[0] and frames.shape[3] == res_curr[1]), f'frames={frames.shape}, res_curr={res_curr}'
            else:
                assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max

        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride}
        return data
    
    def __len__(self):
        return len(self.metadata)

if __name__== "__main__":
    # dataset from config
    # data_config= "myscripts/train_text2video/tv_032_ImgJointTrainDynFPS_2DAE_basedon024/tv_032_ImgJointTrainDynFPS_2DAE_basedon024.yaml"
    # configs = OmegaConf.load(data_config)
    # dataset = instantiate_from_config(configs['data']['params']['train'])
    
    meta_path="/apdcephfs/share_1290939/0_public_datasets/WebVid/metadata/results_2M_train.csv"
    data_dir="/apdcephfs/share_1290939/0_public_datasets/WebVid"
    dataset = WebVid(meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=None,
                 frame_stride=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=True
                 )
    dataloader = DataLoader(dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=True)
    print(f'dataset len={dataset.__len__()}')
    i=0
    total_fps = set()
    res1 = [336,596]
    res2 = [316, 600]
    n_videos_res1=0
    n_videos_res2=0
    n_videos_misc=0
    other_res = []

    for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):
        # pass
        print(f"video={batch['video'].shape}, fps={batch['fps']}")
        total_fps.add(batch['fps'].item())
        if batch['video'].shape[-2] == res1[0] and batch['video'].shape[-1] == res1[1]:
            n_videos_res1 += 1
        elif batch['video'].shape[-2] == res2[0] and batch['video'].shape[-1] == res2[1]:
            n_videos_res2 += 1
        else:
            n_videos_misc += 1
            other_res.append(list(batch['video'].shape[3:]))

        if (i + 1) == 1000:
            break

    print(f'total videos = {i}')
    print('======== total_fps ========')
    print(total_fps)
    print('======== resolution ========')
    print(f'res1 {res1}: n_videos = {n_videos_res1}')
    print(f'res2 {res2}: n_videos = {n_videos_res2}')
    print(f'other resolution: n_videos = {n_videos_misc}')
    print(f'other resolutions: {other_res}')

