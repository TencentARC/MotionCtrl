<p align="center">
  <img src="assets/logo.jpg" height=100>
</p>
<div align="center">

## MotionCtrl: A Unified and Flexible Motion Controller for Video Generation

#### [SIGGRAPH 2024 CONFERENCE PROCEEDINGS]

### <div align="center">👉 MotionCtrl for <b><a href="https://github.com/TencentARC/MotionCtrl/tree/svd">[SVD]</a>, for <a href="https://github.com/TencentARC/MotionCtrl/tree/main">[VideoCrafter]</a>, for <a href="https://github.com/TencentARC/MotionCtrl/tree/animatediff">[AnimateDiff]</a></b></div>

[![Paper](https://img.shields.io/badge/Paper-gray)](https://wzhouxiff.github.io/projects/MotionCtrl/assets/paper/MotionCtrl.pdf) &ensp; [![arXiv](https://img.shields.io/badge/arXiv-red)](https://arxiv.org/pdf/2312.03641.pdf) &ensp; [![Project Page](https://img.shields.io/badge/Project%20Page-green
)](https://wzhouxiff.github.io/projects/MotionCtrl/)

🤗 [![HF Demo](https://img.shields.io/static/v1?label=Demo&message=MotionCtrl%2BSVD&color=orange)](https://huggingface.co/spaces/TencentARC/MotionCtrl_SVD) &ensp; 🤗 [![HF Demo](https://img.shields.io/static/v1?label=Demo&message=MotionCtrl%2BVideoCrafter&color=orange)](https://huggingface.co/spaces/TencentARC/MotionCtrl)

<!-- More examples of MotionCtrl+SVD are in [showcase_svd](https://github.com/TencentARC/MotionCtrl/blob/svd/doc/showcase_svd.md) -->

</div>

---

<!-- https://github.com/TencentARC/MotionCtrl/assets/19488619/45d44bf5-d4bf-4e45-8628-2c8926b5954a -->

---

🔥🔥 We release the codes, [models](https://huggingface.co/TencentARC/MotionCtrl/tree/main) MotionCtrl on [AnimateDiff](https://github.com/guoyww/AnimateDiff).

Official implementation of [MotionCtrl: A Unified and Flexible Motion Controller for Video Generation](https://arxiv.org/abs/2312.03641).

MotionCtrl can Independently control **complex camera motion** and **object motion** of generated videos, with **only a unified** model.

### Results of MotionCtrl+AnimateDiff

<!-- More results are in [showcase_svd](https://github.com/TencentARC/MotionCtrl/blob/svd/doc/showcase_svd.md) and our [Project Page](https://wzhouxiff.github.io/projects/MotionCtrl/). -->

<div align="center">
    <img src="__assets__/demo/sample-d971457c81bca597-Photon.gif", width="600">
    <img src="__assets__/demo/sample-d971457c81bca597-HelloObject.gif", width="600">
    <img src="__assets__/demo/sample-SPIN-LR-HelloObject.gif", width="600">
    <img src="__assets__/demo/sample-none_motion-swaying_0.gif", width="600">
</div>

## ⚙️ Environment
    conda create -n motionctrl python=3.10.6
    conda activate motionctrl
    pip install -r requirements.txt

## 💫 Inference

1. Preparing Models
- [motionctrl_animatediff_cmcm.ckpt](https://huggingface.co/TencentARC/MotionCtrl/blob/main/motionctrl_animatediff_cmcm.ckpt) -> `./models/MotionCtrl`.
- [motionctrl_animatediff_omcm.ckpt](https://huggingface.co/TencentARC/MotionCtrl/blob/main/motionctrl_animatediff_omcm.ckpt) -> `./models/MotionCtrl`.
- [v3_sd15_mm.ckpt.ckpt](https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_mm.ckpt) -> `models/Motion_Module`
- [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) -> `models/StableDiffusion/stable-diffusion-v1-5`
- (Options) LoRAs described in `download_basescripts/`

2. Running scripts:

      - CMCM: `sh scripts/test_cmcm.sh`
      - OMCM: `sh scripts/test_omcm.sh`

## 🔥🔥 Training

1. Preparing Pre-Trained Models
  - [v3_sd15_mm.ckpt.ckpt](https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_mm.ckpt) -> `models/Motion_Module`
  - [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) -> `models/StableDiffusion/stable-diffusion-v1-5`
  
2. Preparing Dataset
  - [RealEstate10K](https://google.github.io/realestate10k/)

    1.  Following https://github.com/cashiwamochi/RealEstate10K_Downloader to download and process the videos.
    2. Corresponding Captions and List are provided in [GoogleDrive](https://drive.google.com/drive/folders/1RYdBW3JUBYMUPdMHL2Yig6getEYN3Z3g?usp=drive_link).
  - [WebVid](https://www.v7labs.com/open-datasets/the-webvid-10m-dataset)
    
    1. Attaining object moving trajectories with [ParticleSfM](https://github.com/bytedance/particle-sfm)
    2. Our collected object trajectories are too large to released. We will show more details about data collection later.

3. Training

    - Train Camera Motion Control: `sh scripts/train_cmcm.sh`
    - Train Object Motion Control:

      - Dense: `sh scripts/train_omcm_dense.sh`
      - Sparse: `sh scripts/train_omcm_sparse.sh`

## :books: Citation
If you make use of our work, please cite our paper.
```bibtex
@inproceedings{wang2024motionctrl,
  title={Motionctrl: A unified and flexible motion controller for video generation},
  author={Wang, Zhouxia and Yuan, Ziyang and Wang, Xintao and Li, Yaowei and Chen, Tianshui and Xia, Menghan and Luo, Ping and Shan, Ying},
  booktitle={ACM SIGGRAPH 2024 Conference Papers},
  pages={1--11},
  year={2024}
}
```

## 🤗 Acknowledgment
The current version of **MotionCtrl** is built on [AnimateDiff](https://github.com/guoyww/AnimateDiff). We appreciate the authors for sharing their awesome codebase.

## ❓ Contact
For any question, feel free to email `wzhoux@connect.hku.hk` or `zhouzi1212@gmail.com`.
