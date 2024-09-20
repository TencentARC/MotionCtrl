# step 1

## start_idx and end_idx is used to process a subset of the dataset in different machines parallelly
python prepare_webvideo_len32.py \
--start_idx 0 \
--end_idx 1000 \

# step 2

root_dir="WebVid/train_256_32"
start_idx=0
end_idx=1000

CUDA_VISIBLE_DEVICES=0 python run_particlesfm_obj_traj.py \
--root_dir $root_dir \
--start_idx $start_idx \
--end_idx $end_idx \