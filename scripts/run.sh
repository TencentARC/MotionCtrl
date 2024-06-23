
# sd + opt
python -m scripts.motionctrl_animate \
    --config configs/prompts/motionctrl/v3-1-RealisticVision_omcm.yaml

# sd + sparse
# python -m scripts.motionctrl_animate \
#     --config configs/prompts/motionctrl/v3-1-sd_omcm_sparse.yaml \
#     --H 512 \
#     --W 512

# sd + sparse + 512
CUDA_VISIBLE_DEVICES=1 python -m scripts.motionctrl_animate \
    --config configs/prompts/motionctrl/v3-1-sd_omcm_sparse_512.yaml \
    --H 512 \
    --W 512

# cmcm + sparse
# CUDA_VISIBLE_DEVICES=1 python -m scripts.motionctrl_animate \
#     --config configs/prompts/motionctrl/v3-1-RealisticVision_omcm_sparse.yaml \
#     --H 512 \
#     --W 512
