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

from gaussiansplatting.scene.cameras import Camera, Simple_Camera
import numpy as np
from gaussiansplatting.utils.general_utils import PILtoTorch
from gaussiansplatting.utils.graphics_utils import fov2focal

from scipy.spatial.transform import Rotation as R, Slerp, RotationSpline
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev, interp1d
from scipy.ndimage import gaussian_filter1d

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def cameraList_load(cam_infos, h, w):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(
            Simple_Camera(colmap_id=c.uid, R=c.R, T=c.T,
                   FoVx=c.FovX, FoVy=c.FovY, h=h, w=w, qvec = c.qvec,
                   image_name=c.image_name, uid=id, data_device='cuda')
        )
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


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

    return [transform_inv @ c2w for c2w in render_poses]


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
    c2ws = []
    for center, rot in zip(all_centers, all_rots):
        c2w = np.eye(4)
        c2w[:3, :3] = rot
        c2w[:3, 3] = center
        c2ws.append(c2w)

    return c2ws