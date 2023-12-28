
<!-- <div align="center"> -->
<!-- <h1>AnimateZero</h1> -->
<!-- <h3><b>MotionCtrl</b>: A Unified and Flexible
                Motion Controller
                for Video Generation</h3> -->

<!-- [![ Paper](https://img.shields.io/badge/Paper-MotionCtrl-red
)](https://wzhouxiff.github.io/projects/MotionCtrl/assets/paper/MotionCtrl.pdf) &ensp; [![ arXiv](https://img.shields.io/badge/arXiv-2312.03641-red
)](https://arxiv.org/pdf/2312.03641.pdf) &ensp; [![Porject Page](https://img.shields.io/badge/Project%20%20Page-MotionCtrl-red)
](https://wzhouxiff.github.io/projects/MotionCtrl/) &ensp; [![Demo](https://img.shields.io/badge/Demo-MotionCtrl-orange
)]() -->

<!-- [![ Paper](https://img.shields.io/badge/Paper-gray
)](https://wzhouxiff.github.io/projects/MotionCtrl/assets/paper/MotionCtrl.pdf) &ensp; [![ arXiv](https://img.shields.io/badge/arXiv-red
)](https://arxiv.org/pdf/2312.03641.pdf) &ensp; [![Porject Page](https://img.shields.io/badge/Project%20Page-green
)
](https://wzhouxiff.github.io/projects/MotionCtrl/) &ensp; [![Demo](https://img.shields.io/badge/Gradio%20Demo-orange
)](https://huggingface.co/spaces/TencentARC/MotionCtrl)

[Zhouxia Wang](https://vvictoryuki.github.io/website/)<sup>1,2</sup>, [Ziyang Yuan](https://github.com/jiangyzy)<sup>1,4</sup>, [Xintao Wang](https://xinntao.github.io/)<sup>1,3</sup>, [Tianshui Chen](http://tianshuichen.com/)<sup>6</sup>, [Menghan Xia](https://menghanxia.github.io/)<sup>3</sup>, [Ping Luo](http://luoping.me/)<sup>2,5</sup>, [Ying Shan](https://scholar.google.com/citations?hl=zh-CN&user=4oXBp9UAAAAJ)<sup>1,3</sup>

<sup>1</sup> ARC Lab, Tencent PCG, <sup>2</sup> The University of Hong Kong, <sup>3</sup> Tencent AI Lab, <sup>4</sup> Tsinghua University, <sup>5</sup> Shanghai AI Laboratory, <sup>6</sup> Guangdong University of Technology


</div> -->

<!-- ## Results of MotionCtrl -->

<!-- </br>
<video poster="" id="steve" autoplay controls muted loop playsinline height="100%" width="100%">
<source src="https://wzhouxiff.github.io/projects/MotionCtrl/assets/videos/teasers/camera_d971457c81bca597.mp4" type="video/mp4">
</video>
<video poster="" id="steve" autoplay controls muted loop playsinline height="100%" width="100%">
<source src="https://wzhouxiff.github.io/projects/MotionCtrl/assets/videos/teasers/camera_Round-R_ZoomIn.mp4" type="video/mp4">
</video>
<video poster="" id="steve" autoplay controls muted loop playsinline height="100%" width="100%">
<source src="https://wzhouxiff.github.io/projects/MotionCtrl/assets/videos/teasers/shake_1.mp4" type="video/mp4">
</video>
<video poster="" id="steve" autoplay controls muted loop playsinline height="100%" width="100%">
<source src="https://wzhouxiff.github.io/projects/MotionCtrl/assets/videos/teasers/s_curve_3_v1.mp4" type="video/mp4">
</video> -->



# MotionCtrl: A Unified and Flexible Motion Controller for Video Generation

[![ Paper](https://img.shields.io/badge/Paper-gray
)](https://wzhouxiff.github.io/projects/MotionCtrl/assets/paper/MotionCtrl.pdf) &ensp; [![ arXiv](https://img.shields.io/badge/arXiv-red
)](https://arxiv.org/pdf/2312.03641.pdf) &ensp; [![Porject Page](https://img.shields.io/badge/Project%20Page-green
)
](https://wzhouxiff.github.io/projects/MotionCtrl/) &ensp; [![ Demo](https://img.shields.io/badge/Gradio%20Demo-orange
)](https://huggingface.co/spaces/TencentARC/MotionCtrl)

---

🔥🔥  This is an official implementation of [MotionCtrl: A Unified and Flexible Motion Controller for Video Generation](https://arxiv.org/pdf/2312.03641.pdf), which is capable of independently controlling the **complex camera motion** and **object motion** of the generated videos, with **only a unified** model. 
There are some results attained with <b>MotionCtrl</b> and more results are showcased in our [Project Page](https://wzhouxiff.github.io/projects/MotionCtrl/).



<div align="center">
    <img src="assets/hpxvu-3d8ym.gif", width="600">
    <img src="assets/w3nb7-9vz5t.gif", width="600">  
    <img src="assets/62n2a-wuvsw.gif", width="600">
    <img src="assets/ilw96-ak827.gif", width="600">
</div>

---

## 📝 Changelog

- [x] 20231225: Release MotionCtrl deployed on ***LVDM/VideoCrafter***.
- [x] 20231225: Gradio Demo Available. [![ Demo](https://img.shields.io/badge/Gradio%20Demo-orange
)](https://huggingface.co/spaces/TencentARC/MotionCtrl)
- [ ] Release MotionCtrl deployed on ***AnimateDiff***.
- [ ] Release MotionCtrl deployed on ***SVD***.

---


## ⚙️ Environment
    conda create -n motionctrl python=3.10.6
    conda activate motionctrl
    pip install -r requirements.txt

## 💫 Inference

1. Download the weights of MotionCtrl [motionctrl.pth](https://huggingface.co/TencentARC/MotionCtrl/blob/main/motionctrl.pth) and put it to `./checkpoints`.
2. Go into `configs/inference/run.sh` and set `condtype` as 'camera_motion', 'object_motion', or 'both'.
- `condtype=camera_motion` means only control the **camera motion** in the generated video.
- `condtype=object_motion` means only control the **object motion** in the generated video.
- `condtype=both` means control the camera motion and object motion in the generated video **simultaneously**.
1. Running scripts:

        sh configs/inference/run.sh




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