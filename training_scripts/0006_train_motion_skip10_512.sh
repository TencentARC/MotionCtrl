
gpus='1,2,3,4,5,6,7'
num_gpus=7

# gpus='1'
# num_gpus=1

CUDA_VISIBLE_DEVICES=$gpus torchrun \
--master_port 1235 \
--nnodes=1 \
--nproc_per_node=$num_gpus \
train_omcm.py \
--config configs/training/v3/0006_cmcm_512_skip10.yaml \
2>&1 | tee -a logs/0006.log

# ,1,2,3,4,5,6,7

# CUDA_VISIBLE_DEVICES=0 torchrun \
# --nnodes=1 \
# --nproc_per_node=1 \
# train.py \
# --config configs/training/training.yaml \
# 2>&1 | tee -a logs/training_v2.log

# CUDA_VISIBLE_DEVICES=0 torchrun \
# --master_port 1234 \
# --nnodes=1 \
# --nproc_per_node=1 \
# train_camera_pose.py \
# --config configs/training/v2/0001_training_camera_pose.yaml \
