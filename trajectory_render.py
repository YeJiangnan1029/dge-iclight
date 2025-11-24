#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import copy
import torch
import numpy as np
from PIL import Image
import mediapy as media
import yaml
from gaussiansplatting.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussiansplatting.gaussian_renderer import render
import torchvision
from gaussiansplatting.scene.camera_scene import CamScene
from gaussiansplatting.utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussiansplatting.scene.gaussian_model import GaussianModel
from gaussiansplatting.utils.graphics_utils import getWorld2View2
from scipy.spatial.transform import Rotation as R, Slerp, RotationSpline
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev, interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_poses_from_cams(views):
    """
    Args:
        views, List[Camera]: 每个 Camera 拥有 .R (3x3) 和 .T (3,)
    Return:
        poses, List[np.ndarray(4,4)]: 每个pose就是w2c矩阵
    """
    poses = []
    for view in views:
        c2w = np.eye(4)
        R_c2w = view.R
        T_c2w = -R_c2w @ view.T
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = T_c2w
        poses.append(np.linalg.inv(c2w))
    return poses
    

def integrate_weights_np(w):
    """Compute the cumulative sum of w, assuming all weight vectors sum to 1.

  The output's size on the last dimension is one greater than that of the input,
  because we're computing the integral corresponding to the endpoints of a step
  function, not the integral of the interior/bin values.

  Args:
    w: Tensor, which will be integrated along the last axis. This is assumed to
      sum to 1 along the last axis, and this function will (silently) break if
      that is not the case.

  Returns:
    cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
  """
    cw = np.minimum(1, np.cumsum(w[..., :-1], axis=-1))
    shape = cw.shape[:-1] + (1,)
    # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
    cw0 = np.concatenate([np.zeros(shape), cw,
                          np.ones(shape)], axis=-1)
    return cw0

def invert_cdf_np(u, t, w_logits):
    """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
    # Compute the PDF and CDF for each weight vector.
    w = np.exp(w_logits) / np.exp(w_logits).sum(axis=-1, keepdims=True)
    cw = integrate_weights_np(w)
    # Interpolate into the inverse CDF.
    interp_fn = np.interp
    t_new = interp_fn(u, cw, t)
    return t_new

def sample_np(rand,
              t,
              w_logits,
              num_samples,
              single_jitter=False,
              deterministic_center=False):
    """
    numpy version of sample()
  """
    eps = np.finfo(np.float32).eps

    # Draw uniform samples.
    if not rand:
        if deterministic_center:
            pad = 1 / (2 * num_samples)
            u = np.linspace(pad, 1. - pad - eps, num_samples)
        else:
            u = np.linspace(0, 1. - eps, num_samples)
        u = np.broadcast_to(u, t.shape[:-1] + (num_samples,))
    else:
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u_max = eps + (1 - eps) / num_samples
        max_jitter = (1 - u_max) / (num_samples - 1) - eps
        d = 1 if single_jitter else num_samples
        u = np.linspace(0, 1 - u_max, num_samples) + \
            np.random.rand(*t.shape[:-1], d) * max_jitter

    return invert_cdf_np(u, t, w_logits)

def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def my_pca(c2ws):
    # 去中心化
    t = c2ws[:, :3, 3]
    t_mean = t.mean(axis=0)
    t_centered = t - t_mean
    # 特征值分解
    cov = np.cov(t_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # 从大到小排序
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    # 保证右手坐标系
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1

    # 4️⃣ 构造变换矩阵 (将原始坐标系变换到PCA坐标系)
    R_pca = eigvecs.T
    T_pca = -R_pca @ t_mean
    transform = np.eye(4)
    transform[:3, :3] = R_pca
    transform[:3, 3] = T_pca

    # 5️⃣ 应用变换
    new_c2ws = []
    for c2w in c2ws:
        new_c2w = transform @ c2w
        new_c2ws.append(new_c2w)
    # new_c2ws = np.stack(new_c2ws, axis=0)

    # 5. position放缩到[-1, 1]
    poses_recentered = np.array(new_c2ws)
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    for c2w in new_c2ws:
        c2w[:3, 3] *= scale_factor

    return np.array(new_c2ws), transform, scale_factor

def my_get_ellipse_path(poses, n_frames=300, z_variation=0., z_phase=0.):
    """
    Docstring for my_get_ellipse_path
    
    :param poses: List[np.array(4,4)], w2c
    :param n_frames: number of frames

    Return List of w2c matrices
    """

    # w2c to c2w
    c2ws = np.array([np.linalg.inv(w2c) for w2c in poses])
    c2ws_recentered, transform, scale_factor = my_pca(c2ws)

    lookat = focus_point_fn(c2ws_recentered)
    lookat_xy = np.array([lookat[0] , lookat[1],  0 ])
    sc = np.percentile(np.abs(c2ws_recentered[:, :3, 3] - lookat_xy), 90, axis=0)
    low = -sc + lookat_xy
    high = sc + lookat_xy
    z_low = np.percentile((c2ws_recentered[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((c2ws_recentered[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            (low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5)),
            (low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5)),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)
    
    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # Resample theta angles so that the velocity is closer to constant.
    lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    theta = sample_np(None, theta, np.log(lengths), n_frames + 1)
    positions = get_positions(theta)
    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = c2ws_recentered[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    render_poses = []
    for p in positions:
        render_pose = np.eye(4)
        forward = normalize(lookat - p)
        right = normalize(np.cross(up, forward))
        up_corrected = normalize(np.cross(forward, right))
        render_pose[:3, :] = np.stack([right, up_corrected, forward, p], axis=1)
        render_poses.append(render_pose)

    for c2w in render_poses:
        c2w[:3, 3] /= scale_factor
    transform_inv = np.linalg.inv(transform)

    return [np.linalg.inv(transform_inv @ c2w) for c2w in render_poses]


def normalize(x):
    return x / np.linalg.norm(x)


def gaussian_smooth_quat(R_list, sigma=1.5):
    q = R.from_matrix(R_list).as_quat()  # (N, 4)
    for i in range(1, len(q)):
        if np.dot(q[i-1], q[i]) < 0:
            q[i] = -q[i]

    # 仅平滑内部帧，保持首尾不动
    q_inner = q[1:-1]
    q_inner_sm = gaussian_filter1d(q_inner, sigma=sigma, axis=0, mode='reflect')
    q_inner_sm /= np.linalg.norm(q_inner_sm, axis=1, keepdims=True)

    q_sm = np.zeros_like(q)
    q_sm[0] = q[0]
    q_sm[-1] = q[-1]
    q_sm[1:-1] = q_inner_sm

    return R.from_quat(q_sm).as_matrix()

def interp_curve(views, smooth=0.05, n_samples=320, show=True, vis_name="vis.png"):
    """
    根据相机外参 (w2c矩阵) 拟合平滑的3D轨迹，并输出对应的平滑旋转矩阵与c2w矩阵。
    
    Args:
        views: List[np.ndarray]，每个元素 (4x4) 相机外参矩阵 (w2c)
        smooth: float, 样条平滑系数，越大曲线越平滑但偏离原点更多
        n_samples: int, 每段输出采样点数量
        show: bool, 是否显示拟合前后对比的3D轨迹
    
    Returns:
        c2w_all: np.ndarray (N_total, 4, 4) — 平滑后的相机外参矩阵（c2w）
    """
    def w2c_to_center_rot(w2c):
        """从 w2c 矩阵中提取相机中心和旋转"""
        c2w = np.linalg.inv(w2c)
        center = c2w[:3, 3]
        rot = c2w[:3, :3]
        return center, rot

    # === 1️⃣ 提取中心与旋转 ===
    centers, R_list = zip(*[w2c_to_center_rot(v) for v in views])
    centers = np.stack(centers, axis=0) # (N, 3)
    R_list = np.stack(R_list, axis=0) 
    quats = R.from_matrix(R_list).as_quat() # (N, 4)
    # 保证四元数连续（防止跳变符号）
    for i in range(1, len(quats)):
        if np.dot(quats[i-1], quats[i]) < 0:
            quats[i] = -quats[i]
    # pose7 = np.hstack([centers, quats])  # (N, 7)

    # === 2️⃣ 检测跳跃点，分段 ===
    d = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    thresh = np.mean(d) + 3 * np.std(d)
    mask = d < thresh
    min_len = 10
    split_indices = []
    N = len(mask)
    start = None
    for i in range(N):
        if mask[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_len:
                    split_indices.append(np.arange(start, i))
                start = None
    # 最后一段
    if start is not None and N - start >= min_len:
        split_indices.append(np.arange(start, N))

    # === 3️⃣ 对每段独立插值 ===
    all_centers, all_rots, all_seg_ids = [], [], []
    seg_num = len(split_indices)
    for seg in split_indices:
        if len(seg) < 3:
            continue  # 样本太少无法拟合
        C_seg = centers[seg]
        R_seg = R_list[seg]
        # pose7_seg = pose7[seg]

        # 弦长参数化
        seglen = np.linalg.norm(np.diff(C_seg, axis=0), axis=1)
        u = np.concatenate(([0.0], np.cumsum(seglen)))
        u /= u[-1]

        # 样条拟合位置
        tck, _ = splprep(C_seg.T, u=u, s=smooth)

        # 等弧长重采样
        u_dense = np.linspace(0, 1, 2000)
        p_dense = np.vstack(splev(u_dense, tck)).T
        seg = np.linalg.norm(np.diff(p_dense, axis=0), axis=1)
        arc = np.concatenate(([0], np.cumsum(seg)))
        arc /= arc[-1]
        interp_u = np.interp(np.linspace(0, 1, int(n_samples/seg_num)), arc, u_dense)
        interp_centers = np.vstack(splev(interp_u, tck)).T

        # 旋转插值
        R_seg = gaussian_smooth_quat(R_seg, sigma=3.0)
        rot_obj = R.from_matrix(R_seg)
        rot_spline = RotationSpline(u, rot_obj)
        interp_rots = rot_spline(interp_u).as_matrix()

        # 保存结果
        all_centers.append(interp_centers)
        all_rots.append(interp_rots)

    # === 4️⃣ 拼接所有段 ===
    all_centers = np.concatenate(all_centers, axis=0)
    all_rots = np.concatenate(all_rots, axis=0)

    # === 5️⃣ 构建 c2w 矩阵 ===
    w2cs = []
    for center, rot in zip(all_centers, all_rots):
        c2w = np.eye(4)
        c2w[:3, :3] = rot
        c2w[:3, 3] = center
        w2cs.append(np.linalg.inv(c2w))

    return w2cs

def render_views(source_path: str, gs_source : str, exp_folder: str, pipeline : PipelineParams, scene_key, output_path):
    with torch.no_grad():
        gaussians = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2
        )
        gaussians.load_ply(gs_source)
        gaussians.localize = False
        cam_scene = CamScene(source_path, h=512, w=512, transforms_path=exp_folder, image_limit=-1)
        train_views = cam_scene.cameras
        view = copy.deepcopy(train_views[0])
        training_poses = get_poses_from_cams(train_views)
        # visualize_camera_path(training_poses, stride=5, filename=f"{scene_key.replace('/', '_')}_training.png")
        if "mip360" in scene_key:
            render_poses = my_get_ellipse_path(training_poses, 320, z_variation=0.1)
        elif "scannetpp" in scene_key:
            render_poses = my_get_ellipse_path(training_poses, 320, z_variation=0.1)
            # render_poses = generate_interp_path(training_poses, smooth=0.02, n_samples=320)
        else:
            render_poses = interp_curve(training_poses, smooth=0.3, n_samples=320)
        
        # visualize_camera_path(render_poses, stride=1, arrow_length=0.1, filename=f"{scene_key.replace('/', '_')}_trajectory.png")
        # render_poses, _ = interp_curve(training_poses, vis_name=f"{scene_key}.png")
        
        bg_color = [1,1,1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # render_path = Path("temp/trajectory_vis")
        output_video_path = Path(output_path) / "edited_trajectory.mp4"
        # visualize_camera_path(training_poses, stride=5, filename=filename)
        frames = []
        for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
            view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
            view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            view.camera_center = view.world_view_transform.inverse()[3, :3]
            rendering = render(view, gaussians, pipeline, background)
            img = torch.clamp(rendering["render"], min=0., max=1.)
            img_np = img.detach().cpu().permute(1, 2, 0).numpy()  # [0, 1] float
            frames.append(img_np)

        media.write_video(output_video_path, frames, fps=32)  # 你可以根据需要修改fps
        print(f"Video saved to: {output_video_path}")


def visualize_camera_path(render_poses, stride=10, arrow_length=0.1, filename="cams_vis.png"):
    """
    可视化相机轨迹与朝向
    Args:
        render_poses: List[np.ndarray], 每个为 4x4 位姿矩阵 (world←camera)
        stride: 每隔多少帧绘制一个箭头
        arrow_length: 箭头长度比例
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 提取相机中心与朝向
    cam_centers = []
    cam_dirs = []
    for pose in render_poses:
        c2w = np.linalg.inv(pose)
        cam_center = c2w[:3, 3]
        cam_dir = c2w[:3, 2]  # z轴朝向
        cam_centers.append(cam_center)
        cam_dirs.append(cam_dir)
    cam_centers = np.stack(cam_centers)
    cam_dirs = np.stack(cam_dirs)

    # 绘制轨迹
    ax.scatter(cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2], 'b-', label='Camera Path')

    # 绘制相机方向箭头
    for i in range(0, len(cam_centers), stride):
        c = cam_centers[i]
        d = cam_dirs[i]
        ax.quiver(c[0], c[1], c[2], d[0], d[1], d[2],
                  length=arrow_length, color='r', normalize=True)

    # 绘制起点与终点
    ax.scatter(cam_centers[0, 0], cam_centers[0, 1], cam_centers[0, 2], c='g', s=60, label='Start')
    ax.scatter(cam_centers[-1, 0], cam_centers[-1, 1], cam_centers[-1, 2], c='orange', s=60, label='End')

    # 美化
    ax.legend()
    ax.set_title("Camera Trajectory Visualization")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"temp/trajectory_vis/{filename}")

# if __name__ == "__main__":
#     # Set up command line argument parser
    # parser = ArgumentParser(description="Testing script parameters")
#     model = ModelParams(parser, sentinel=True)
    # pipeline = PipelineParams(parser)
#     parser.add_argument("--quiet", action="store_true")
#     parser.add_argument("--gs-source", type=str, required=True)
#     parser.add_argument("--exp-folder", type=str, default="")
#     args = get_combined_args(parser)
#     print("[Warn] We ignore the model folder path")
#     print("Rendering " +  args.gs_source)

#     # Initialize system state (RNG)
#     safe_state(args.quiet)

#     render_sets(model.extract(args).source_path, args.gs_source, args.exp_folder, pipeline.extract(args))

parser = ArgumentParser(description="Testing script parameters")
pipeline = PipelineParams(parser)

source_config = Path("script/benchmark/data_source.yaml")
output_folder = Path("temp/trajectory_vis")
os.makedirs(output_folder, exist_ok=True)

print("Loading data source config:", source_config)
with open(source_config, "r") as f:
    data_sources = yaml.safe_load(f)

task_base_folder = Path("/mnt/16T/yejiangnan/work/cvpr25_EditSplat/output/editsplat_output")

for dataset_name, dataset_info in data_sources.items():
    # if dataset_name != 'scannetpp': continue
    scenes = dataset_info.get("scenes", [])
    base_path = Path(dataset_info['base_path'])
    gs_base_path = Path(dataset_info['gs_base_path'])
    gs_ply = Path(dataset_info['gs_ply'])
    exp_folder = dataset_info['exp_folder'] if "exp_folder" in dataset_info else ""

    print(f"\nProcessing dataset: {dataset_name}, total scenes: {len(scenes)}")

    for scene_name in tqdm(scenes):
        scene_key = f"{dataset_name}/{scene_name}"

        scene_subfolder = dataset_info.get("scene_subfolder", None)
        if scene_subfolder:
            scene_full = f"{scene_name}/{scene_subfolder}"
        else:
            scene_full = scene_name
        
        data_source = base_path / scene_full
        gs_source = gs_base_path / scene_name / gs_ply
        exp_path = gs_base_path / scene_full / exp_folder if exp_folder else ""

        for idx in range(3):
            task_name = f"editsplat_{dataset_name}_{scene_name}_{idx}"
            task_folder = task_base_folder / task_name

            gs_source = task_folder / "point_cloud/iteration_30650/point_cloud.ply"

            if not gs_source.exists() or (Path(task_folder) / "edited_trajectory.mp4").exists():
                print(f"Skipping {task_name}, either gs_source missing or video already exists.")
                continue
            
            render_views(data_source, gs_source, exp_path, pipeline, scene_key.replace('/', '_'), task_folder)
