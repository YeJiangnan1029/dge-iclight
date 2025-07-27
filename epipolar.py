import os
import torch
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

def compute_camera_distance(cams, key_cams):
    cam_centers = [cam.camera_center for cam in cams]
    key_cam_centers = [cam.camera_center for cam in key_cams] 
    cam_centers = torch.stack(cam_centers).cuda()
    key_cam_centers = torch.stack(key_cam_centers).cuda()
    cam_distance = torch.cdist(cam_centers, key_cam_centers)

    return cam_distance   

np.random.seed(42)
key_cams = torch.load("temp/key_cams.pkl")
cams = torch.load("temp/cams.pkl")
epipolar_constrains = torch.load("temp/epipolar_constrains.pkl")
image_path = "edit_cache/data-dge_data-face-scene_point_cloud.ply/origin_render"

norm_hidden_states = torch.load("temp/norm_hidden_states.pkl")
pivot_hidden_states = torch.load("temp/pivot_hidden_states.pkl")

_, n_frames, sequence_length, dim = norm_hidden_states.shape

# images = {}
# for idx, cam in enumerate(cams):
#     img_id = cam.uid
#     image_file = f"{image_path}/{img_id:04}.png"
#     image = Image.open(image_file)
#     image.save(f"temp/vis/cam{idx}_id{img_id}.png")
#     images[idx] = image
    
# key_images = {}
# for idx, cam in enumerate(key_cams):
#     img_id = cam.uid
#     image_file = f"{image_path}/{img_id:04}.png"
#     image = Image.open(image_file)
#     image.save(f"temp/vis/key_cam{idx}_id{img_id}.png")
#     key_images[idx] = image


idx1 = []
idx2 = []
cam_distance = compute_camera_distance(cams, key_cams)
cam_distance_min = cam_distance.sort(dim=-1)
closest_cam = cam_distance_min[1][:,:1]
closest_cam_pivot_hidden_states = pivot_hidden_states[1][closest_cam]
sim = torch.einsum('bld,bcsd->bcls', norm_hidden_states[1] / norm_hidden_states[1].norm(dim=-1, keepdim=True), closest_cam_pivot_hidden_states / closest_cam_pivot_hidden_states.norm(dim=-1, keepdim=True)).squeeze()
sim = sim.view(-1, sequence_length)
sim_max = sim.max(dim=-1)
idx1.append(sim_max[1])

idx1 = []
pivot_this_batch = 0

idx1_epipolar = epipolar_constrains[sequence_length].gather(dim=1, index=closest_cam[:, :, None, None].expand(-1, -1, epipolar_constrains[sequence_length].shape[2], epipolar_constrains[sequence_length].shape[3])).cuda()

idx1_epipolar = idx1_epipolar.view(n_frames, -1, sequence_length)
idx1_epipolar[pivot_this_batch, ...] = False

idx1_epipolar = idx1_epipolar.view(n_frames * sequence_length, sequence_length)
idx1_sum = idx1_epipolar.sum(dim=-1)
idx1_epipolar[idx1_sum == sequence_length, :] = False
sim[idx1_epipolar] = 0
sim_max = sim.max(dim=-1)

idx1 = sim_max[1].view(n_frames, -1)

sim = sim.reshape(n_frames, -1, sequence_length)
res = 64
key_id = 0
key_cam = key_cams[key_id]
for cam_id, cam in enumerate(cams):
    # for key_id, key_cam in enumerate(key_cams):
    n_tokens = 10
    random_tokens = np.random.choice(res*res, n_tokens, replace=False)
    for token_id in random_tokens:
        epipolar_bool = epipolar_constrains[res*res][cam_id, key_id, token_id].reshape(res, res)
        # epipolar_bool = compute_epipolar_constrains(key_cam, cam, current_H=res, current_W=res)[token_id].reshape(res, res)
        epipolar_bool = epipolar_bool.cpu().numpy().astype(bool)
        image_1 = np.array(Image.open(f"{image_path}/{key_cam.uid:04}.png").resize((res, res)))
        image_2 = np.array(Image.open(f"{image_path}/{cam.uid:04}.png").resize((res, res)))
        row, col = divmod(token_id, res)
        match_token_id = idx1[cam_id, token_id].item()
        mrow, mcol = divmod(match_token_id, res)
        
        sim_heatmap = sim[cam_id, token_id].reshape(res, res).cpu().numpy()
        
        fig, axs = plt.subplots(2, 2, figsize=(8, 4))
        
        axs[0][0].imshow(image_1)
        axs[0][0].scatter([mcol], [mrow], c="red", s=40, marker="o")
        # axs[0].imshow(sim_heatmap, cmap="jet", alpha=0.5)  # 半透明叠加
        axs[0][0].imshow(epipolar_bool, alpha=0.45)   # 半透明叠加
        axs[0][0].set_title(f"Key-cam {key_cam.uid}: epipolar line")
        axs[0][0].axis("off")

        # 右：极线掩码
        axs[0][1].imshow(image_2)
        axs[0][1].scatter([col], [row], c="red", s=40, marker="o")
        # axs[1].imshow(epipolar_bool, alpha=0.45)   # 半透明叠加
        axs[0][1].set_title(f"Cam {cam.uid}: token {token_id}")
        axs[0][1].axis("off")
        
        axs[1][0].imshow(sim_heatmap, cmap="jet")  # 半透明叠加
        axs[1][0].set_title(f"Key-cam {key_cam.uid}: epipolar line")
        axs[1][0].axis("off")
        
        axs[1][1].imshow(image_2)
        axs[1][1].scatter([col], [row], c="red", s=40, marker="o")
        # axs[1].imshow(epipolar_bool, alpha=0.45)   # 半透明叠加
        axs[1][1].set_title(f"Cam {cam.uid}: token {token_id}")
        axs[1][1].axis("off")

        plt.tight_layout()

        # 保存 / 显示
        out_file = f"temp/match/epipolar_cam{cam.uid}_key{key_cam.uid}_token{token_id}.png"
        plt.savefig(out_file, dpi=200)
        print(f"saved: {out_file}")


        plt.close(fig)