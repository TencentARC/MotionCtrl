

ckpt='checkpoints/motionctrl_svd.ckpt'
config='configs/inference/config_motionctrl_cmcm.yaml'

height=576
width=1024
cond_aug=0.02

fps=10

image_input='examples/basic/eduardo-gorghetto-5auIBbcoRNw-unsplash.jpg'

res_dir="outputs/motionctrl_svd"
if [ ! -d $res_dir ]; then
    mkdir -p $res_dir
fi

python main/inference/motionctrl_cmcm.py \
--seed 12345 \
--ckpt $ckpt \
--config $config \
--savedir $res_dir \
--savefps 10 \
--ddim_steps 25 \
--frames 14 \
--input $image_input \
--fps $fps \
--motion 127 \
--cond_aug $cond_aug \
--decoding_t 1 --resize \
--height $height --width $width \
--sample_num 2 \
--transform \
--pose_dir 'examples/camera_poses' \
--speed 2.0 \