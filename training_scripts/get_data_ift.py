import os, sys
import json
from tqdm import tqdm

data_dir = '/group/40046/public_datasets/3d_datasets/objaverse/views_release'
info_file = 'objaverse_success_caption.json'

with open(os.path.join(data_dir, info_file)) as f:
    captions = json.load(f)

metadata = list(captions.keys())
print('============= length of dataset %d =============' % len(metadata))
print('metadata[0-10]:', metadata[:10])

start_idx = int(sys.argv[1])
get_len = int(sys.argv[2])

for index in tqdm(range(start_idx, start_idx+get_len)):
    key = metadata[index]
    data_path = os.path.join(data_dir, 'views_circular', key)
    cmd = f'ft put --skip -s work_mac_new/objaverse/views_circular/ {data_path}'
    os.system(cmd)