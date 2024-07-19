
gpus='0,1,2,3,4,5,6,7'
num_gpus=8

# For debugging
gpus='1'
num_gpus=1

CUDA_VISIBLE_DEVICES=$gpus torchrun \
--master_port 1235 \
--nnodes=1 \
--nproc_per_node=$num_gpus \
train.py \
--config configs/training/cmcm.yaml
