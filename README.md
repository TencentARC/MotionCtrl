# MotionCtrl: A Unified and Flexible Motion Controller for Video Generation

[![ Paper](https://img.shields.io/badge/Paper-gray
)](https://wzhouxiff.github.io/projects/MotionCtrl/assets/paper/MotionCtrl.pdf) &ensp; [![ arXiv](https://img.shields.io/badge/arXiv-red
)](https://arxiv.org/pdf/2312.03641.pdf) &ensp; [![Porject Page](https://img.shields.io/badge/Project%20Page-green
)
](https://wzhouxiff.github.io/projects/MotionCtrl/) &ensp; [![ Demo](https://img.shields.io/badge/Demo%3AMotionCtrl%2BVideoCrafter-orange
)](https://huggingface.co/spaces/TencentARC/MotionCtrl) &ensp; [![ Demo](https://img.shields.io/badge/Demo%3AMotionCtrl%2BSVD-orange
)](https://huggingface.co/spaces/TencentARC/MotionCtrl_SVD).

---

🔥🔥  This is an official implementation of [MotionCtrl: A Unified and Flexible Motion Controller for Video Generation](https://arxiv.org/pdf/2312.03641.pdf), which is capable of independently controlling the **complex camera motion** and **object motion** of the generated videos, with **only a unified** model. 
There are some results attained with <b>MotionCtrl</b> and more results are showcased in our [Project Page](https://wzhouxiff.github.io/projects/MotionCtrl/).

### Results of MotionCtrl+SVD
<div align="center">
    <img src="assets/svd/00_ibzz5-dxv2h.gif", width="300">
    <img src="assets/svd/01_5guvn-0x6v2.gif", width="300">  
    <!-- <img src="assets/svd/10_inrmo-e2o0q.gif", width="300">
    <img src="assets/svd/11_2lfsc-m217n.gif", width="300"> -->
    <img src="assets/svd/12_sn7bz-0hcaf.gif", width="300">
    <img src="assets/svd/13_3lyco-4ru8j.gif", width="300">
</div>

### Results of MotionCtrl+VideoCrafter
<div align="center">
    <img src="assets/hpxvu-3d8ym.gif", width="600">
    <img src="assets/w3nb7-9vz5t.gif", width="600">  
    <img src="assets/62n2a-wuvsw.gif", width="600">
    <img src="assets/ilw96-ak827.gif", width="600">
</div>

---

## 📝 Changelog

- [x] 20231225: Release MotionCtrl deployed on ***LVDM/VideoCrafter***.
- [x] 20231225: Gradio Demo Available. [![ Demo](https://img.shields.io/badge/Demo%3AMotionCtrl%2BVideoCrafter-orange
)](https://huggingface.co/spaces/TencentARC/MotionCtrl)
- [x] 20231228: Provide local gradio demo for convenience.
- [x] 20240115 More camera poses used for testing are provided in `dataset/camera_poses`
- [ ] 20240115 Release MotionCtrl deployed on ***SVD***. Code will be in brach [MotionCtrl_SVD](https://github.com/TencentARC/MotionCtrl/tree/MotionCtrl_SVD) and Gradio Demo will be available in [![ Demo](https://img.shields.io/badge/Demo%3AMotionCtrl%2BSVD-orange
)](https://huggingface.co/spaces/TencentARC/MotionCtrl_SVD). 
- [ ] Release MotionCtrl deployed on ***AnimateDiff***.


---


## ⚙️ Environment
    conda create -n motionctrl python=3.10.6
    conda activate motionctrl
    pip install -r requirements.txt

## 💫 Inference

- #### Run local inference script

1. Download the weights of MotionCtrl [motionctrl.pth](https://huggingface.co/TencentARC/MotionCtrl/blob/main/motionctrl.pth) and put it to `./checkpoints`.
2. Go into `configs/inference/run.sh` and set `condtype` as 'camera_motion', 'object_motion', or 'both'.
- `condtype=camera_motion` means only control the **camera motion** in the generated video.
- `condtype=object_motion` means only control the **object motion** in the generated video.
- `condtype=both` means control the camera motion and object motion in the generated video **simultaneously**.
3. Running scripts:
        sh configs/inference/run.sh

- #### Run local gradio demo
      python -m app --share



## :books: Citation
If you make use of our work, please cite our paper.
```bibtex
@inproceedings{wang2023motionctrl,
  title={MotionCtrl: A Unified and Flexible Motion Controller for Video Generation},
  author={Wang, Zhouxia and Yuan, Ziyang and Wang, Xintao and Chen, Tianshui and Xia, Menghan and Luo, Ping and Shan, Yin},
  booktitle={arXiv preprint arXiv:2312.03641},
  year={2023}
}
```

## 🤗 Acknowledgment
The current version of **MotionCtrl** is built on [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter). We appreciate the authors for sharing their awesome codebase.

## ❓ Contact
For any question, feel free to email `wzhoux@connect.hku.hk` or `zhouzi1212@gmail.com`.
