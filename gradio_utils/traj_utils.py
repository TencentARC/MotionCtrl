import cv2
import numpy as np

from gradio_utils.flow_utils import bivariate_Gaussian

OBJECT_MOTION_MODE = ["Provided Trajectory", "Custom Trajectory"]

PROVIDED_TRAJS = {
    "horizon_1": "examples/trajectories/horizon_2.txt",
    "swaying_1": "examples/trajectories/shake_1.txt",
    "swaying_2": "examples/trajectories/shake_2.txt",
    "swaying_3": "examples/trajectories/shaking_10.txt",
    "curve_1": "examples/trajectories/curve_1.txt",
    "curve_2": "examples/trajectories/curve_2.txt",
    "curve_3": "examples/trajectories/curve_3.txt",
    "curve_4": "examples/trajectories/curve_4.txt",
}


def read_points(file, video_len=16, reverse=False):
    with open(file, 'r') as f:
        lines = f.readlines()
    points = []
    for line in lines:
        x, y = line.strip().split(',')
        points.append((int(x), int(y)))
    if reverse:
        points = points[::-1]

    if len(points) > video_len:
        skip = len(points) // video_len
        points = points[::skip]
    points = points[:video_len]
    
    return points

def get_provided_traj(traj_name):
    traj = read_points(PROVIDED_TRAJS[traj_name])
    # xrange from 256 to 1024
    traj = [[int(1024*x/256), int(1024*y/256)] for x,y in traj]
    return traj

blur_kernel = bivariate_Gaussian(99, 10, 10, 0, grid=None, isotropic=True)

def process_points(points):
    frames = 16
    defualt_points = [[512,512]]*16

    if len(points) < 2:
        return defualt_points
    elif len(points) >= frames:
        skip = len(points)//frames
        return points[::skip][:15] + points[-1:]
    else:
        insert_num = frames - len(points)
        insert_num_dict = {}
        interval = len(points) - 1
        n = insert_num // interval
        m = insert_num % interval
        for i in range(interval):
            insert_num_dict[i] = n
        for i in range(m):
            insert_num_dict[i] += 1

        res = []
        for i in range(interval):
            insert_points = []
            x0,y0 = points[i]
            x1,y1 = points[i+1]

            delta_x = x1 - x0
            delta_y = y1 - y0
            for j in range(insert_num_dict[i]):
                x = x0 + (j+1)/(insert_num_dict[i]+1)*delta_x
                y = y0 + (j+1)/(insert_num_dict[i]+1)*delta_y
                insert_points.append([int(x), int(y)])

            res += points[i:i+1] + insert_points
        res += points[-1:]
        return res

def get_flow(points, video_len=16):
    optical_flow = np.zeros((video_len, 256, 256, 2), dtype=np.float32)
    for i in range(video_len-1):
        p = points[i]
        p1 = points[i+1]
        optical_flow[i+1, p[1], p[0], 0] = p1[0] - p[0]
        optical_flow[i+1, p[1], p[0], 1] = p1[1] - p[1]
    for i in range(1, video_len):
        optical_flow[i] = cv2.filter2D(optical_flow[i], -1, blur_kernel)


    return optical_flow


def process_traj(points, device='cpu'):
    xy_range = 1024
    points = process_points(points)
    points = [[int(256*x/xy_range), int(256*y/xy_range)] for x,y in points]
    
    optical_flow = get_flow(points)
    # optical_flow = torch.tensor(optical_flow).to(device)

    return optical_flow