CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun \
--master_port 1234 \
--nnodes=1 \
--nproc_per_node=7 \
train_camera_pose.py \
--config configs/training/v2/training.yaml \
# 2>&1 | tee -a logs/training_v2.log


# CUDA_VISIBLE_DEVICES=0 torchrun \
# --nnodes=1 \
# --nproc_per_node=1 \
# train.py \
# --config configs/training/training.yaml \
# 2>&1 | tee -a logs/training_v2.log
