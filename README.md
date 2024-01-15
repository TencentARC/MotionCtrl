# MotionCtrl: A Unified and Flexible Motion Controller for Video Generation

[![ Paper](https://img.shields.io/badge/Paper-gray
)](https://wzhouxiff.github.io/projects/MotionCtrl/assets/paper/MotionCtrl.pdf) &ensp; [![ arXiv](https://img.shields.io/badge/arXiv-red
)](https://arxiv.org/pdf/2312.03641.pdf) &ensp; [![Porject Page](https://img.shields.io/badge/Project%20Page-green
)
](https://wzhouxiff.github.io/projects/MotionCtrl/) &ensp; [![ Demo](https://img.shields.io/badge/Demo%3AMotionCtrl%2BVideoCrafter-orange
)](https://huggingface.co/spaces/TencentARC/MotionCtrl) &ensp; [![ Demo](https://img.shields.io/badge/Demo%3AMotionCtrl%2BSVD-orange
)](https://huggingface.co/spaces/TencentARC/MotionCtrl_SVD).

---

## MotionCtrl + SVD 

🔥🔥  This is an extension of [MotionCtrl: A Unified and Flexible Motion Controller for Video Generation](https://arxiv.org/pdf/2312.03641.pdf), deployed on [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid). Current version is capable of controlling both **basic and complex camera motion** of the generated videos, with **only a unified** model.

🔥🔥 Corresponding gradio demo [![ Demo](https://img.shields.io/badge/Demo%3AMotionCtrl%2BSVD-orange
)](https://huggingface.co/spaces/TencentARC/MotionCtrl_SVD) is available now. [Quick start]() is provided in `doc/quick.md`. There are some results.

<div align="center">
    <img src="assets/svd/00_ibzz5-dxv2h.gif", width="300">
    <img src="assets/svd/01_5guvn-0x6v2.gif", width="300">  
    <img src="assets/svd/12_sn7bz-0hcaf.gif", width="300">
    <img src="assets/svd/10_inrmo-e2o0q.gif", width="300">
    <img src="assets/svd/11_2lfsc-m217n.gif", width="300">
    <img src="assets/svd/13_3lyco-4ru8j.gif", width="300">
</div>

## ⚙️ Environment
    conda create -n motionctrl python=3.10.6
    conda activate motionctrl
    pip install -r requirements.txt

## 💫 Inference

- #### Run local inference script

1. Download the weights of MotionCtrl [motionctrl_svd.pth](https://huggingface.co/TencentARC/MotionCtrl/blob/main/motionctrl_svd.ckpt) and put it to `./checkpoints`.
2. Running scripts:
        sh configs/inference/run.sh

- #### Run local gradio demo
      python -m app --share

❗❗❗ **Noted** ❗❗❗
1. If the motion control is not obvious, try to increase the `speed` in the run.sh or `Motion Speed` in the gradio demo.
2. If the generated videos are distored severely, try to descrease the `speed` in the run.sh or `Motion Speed` in the gradio demo. Or increase `FPS`.


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
The current version of **MotionCtrl** is built on [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid). We appreciate the authors for sharing their awesome codebase.

## ❓ Contact
For any question, feel free to email `wzhoux@connect.hku.hk` or `zhouzi1212@gmail.com`.
