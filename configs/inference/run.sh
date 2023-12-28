
config="configs/inference/config_both.yaml"
ckpt='./checkpoints/motionctrl.pth'

condtype='both'
condtype='object_motion'
condtype='camera_motion'

cond_dir="examples/"

res_dir="./outputs/"
if [ ! -d $res_dir ]; then
    mkdir -p $res_dir
fi

save_dir=$res_dir/$condtype'_seed'$seed

use_ddp=0

if [ $use_ddp == 0 ]; then
python 'main/evaluation/motionctrl_inference.py' \
--seed 1234 \
--ckpt_path $ckpt \
--base $config \
--savedir $save_dir \
--n_samples 5 \
--bs 1 --height 256 --width 256 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--condtype $condtype \
--cond_dir $cond_dir \
# --save_imgs
fi

if [ $use_ddp == 1 ]; then
python3 -m torch.distributed.launch \
--nproc_per_node=3 --nnodes=1 --master_port=23466 \
main/evaluation/ddp_wrapper.py \
--module 'inference' \
--seed 2000 \
--ckpt_path $ckpt \
--base $config \
--savedir $res_dir/$name \
--n_samples 3 \
--bs 1 --height 256 --width 256 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--condtype $condtype \
--cond_dir $cond_dir
fi