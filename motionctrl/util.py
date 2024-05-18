import torch
from einops import rearrange
from decord import VideoReader, cpu
from motionctrl.flow_viz import flow_to_image
import numpy as np

def get_traj_features(trajs, omcm):
    b, c, f, h, w = trajs.shape
    trajs = rearrange(trajs, "b c f h w -> (b f) c h w")
    traj_features = omcm(trajs)
    traj_features = [rearrange(traj_feature, "(b f) c h w -> b c f h w", f=f) for traj_feature in traj_features]

    return traj_features

def get_batch_motion(opt_model, num_reg_refine, batch_x, t=None):
    # batch_x: [-1, 1], [b, c, t, h, w]
    # input of motion model should be in range [0, 255]
    # output of motion model [B, 2, t, H, W]

    # batch_x [B, c, t, h, w]
    # b, c, t, h, w = batch_x.shape
    t = t if t is not None else batch_x.shape[2]
    batch_x = (batch_x + 1) * 0.5 * 255.

    motions = []
    for i in range(t-1):
        image1 = batch_x[:, :, i]
        image2 = batch_x[:, :, i+1]

        with torch.no_grad():
            results_dict = opt_model(image1, image2,
                                    attn_type='swin',
                                    attn_splits_list=[2, 8],
                                    corr_radius_list=[-1, 4],
                                    prop_radius_list=[-1, 1],
                                    num_reg_refine=num_reg_refine,
                                    task='flow',
                                    pred_bidir_flow=False,
                                    )
        motions.append(results_dict['flow_preds'][-1]) # [B, 2, H, W]

    motions = [torch.zeros_like(motions[0])] + motions # append a zero motion for the first frame
    motions = torch.stack(motions, dim=2) # [B, 2, t, H, W]

    return motions

def get_opt_from_video(opt_model, num_reg_refine, video_path, width, height, num_frames, device):
    video_reader = VideoReader(str(video_path), ctx=cpu(0),
                                   width=width, height=height)
    fps_ori = video_reader.get_avg_fps()
    frame_stride = len(video_reader) // num_frames
    frame_stride = min(frame_stride, 4)

    frame_indices = [frame_stride*i for i in range(num_frames)]
    video_data = video_reader.get_batch(frame_indices).asnumpy()
    video_data = torch.Tensor(video_data).permute(3, 0, 1, 2).float() # [c, t, h, w]
    video_data = video_data / 255.0 * 2.0 - 1.0
    # video_data = video_data.unsqueeze(0).repeat(2, 1, 1, 1, 1) # [2, c, t, h, w]
    video_data = video_data.unsqueeze(0) # [1, c, t, h, w]
    video_data = video_data.to(device)
    optcal_flow = get_batch_motion(opt_model, num_reg_refine, video_data, t=num_frames)

    return optcal_flow

def vis_opt_flow(flow):
    # flow: [b c t h w]
    vis_flow = []

    for i in range(0, flow.shape[2]):
        cur_flow = flow[0, :, i].permute(1, 2, 0).data.cpu().numpy()
        cur_flow = flow_to_image(cur_flow)
        vis_flow.append(cur_flow)
    vis_flow = np.stack(vis_flow, axis=0)
    vis_flow = vis_flow[:, :, :, ::-1] # [t, h, w, c]
    vis_flow = vis_flow / 255.0 # [0, 1]
    # [t, h, w, c] -> [c, t, h, w]
    vis_flow = rearrange(vis_flow, "t h w c -> c t h w")
    vis_flow = torch.Tensor(vis_flow) # [c, t, h, w]
    vis_flow = vis_flow[None, ...]

    return vis_flow