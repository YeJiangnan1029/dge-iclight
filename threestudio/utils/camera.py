import torch

from gaussiansplatting.utils.graphics_utils import fov2focal
from gaussiansplatting.scene.cameras import Simple_Camera
import torch.nn.functional as F
from threestudio.utils.typing import *
import numpy as np

import numpy as np

def get_spiral_path(cameras, center, radius, up, frames=200, n_rot=1):
    """
    绕场景中心的水平环绕轨迹（无上下浮动），
    lookAt = center 在所有输入 camera 中轴线上的投影点的均值。
    
    Args:
        cameras: list，相机对象，需包含 world_view_transform
        center: ndarray (3,) 原始场景中心
        radius: float 环绕半径
        up: ndarray (3,) 上方向
        frames: int, 帧数
        n_rot: int, 旋转圈数

    Returns:
        list[np.ndarray]，每个 4x4 c2w 矩阵
    """

    proj_points = []

    # --- 1) 遍历每个相机，将 center 投影到相机中轴线上
    for cam in cameras:
        c2w = np.linalg.inv(cam.world_view_transform.T.cpu().numpy())
        cam_center = c2w[:3, 3]
        cam_forward = c2w[:3, 2]  # 第三列为 forward
        cam_forward /= np.linalg.norm(cam_forward)

        v = center - cam_center
        alpha = np.dot(v, cam_forward)  # 投影长度
        proj_point = cam_center + alpha * cam_forward
        proj_points.append(proj_point)

    proj_points = np.stack(proj_points, axis=0)
    look_at = proj_points.mean(axis=0)  # --- 新的 lookAt

    # --- 2) 归一化 up
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-8:
        up = np.array([0.0, 1.0, 0.0])
    else:
        up = up / up_norm

    # --- 3) 构建水平基：取第一个相机到 lookAt 的向量投影到水平面
    cam0_center = np.linalg.inv(cameras[0].world_view_transform.T.cpu().numpy())[:3, 3]
    init_dir = cam0_center - look_at
    proj = init_dir - np.dot(init_dir, up) * up  # 投影到水平面
    proj_norm = np.linalg.norm(proj)
    if proj_norm < 1e-6:
        tmp = np.array([0.0, 1.0, 0.0]) if abs(up[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        proj = tmp - np.dot(tmp, up) * up
        proj /= np.linalg.norm(proj)
    else:
        proj /= proj_norm

    forward0 = proj
    right0 = np.cross(forward0, up)
    right0 /= np.linalg.norm(right0)

    # --- 4) 沿圆生成轨迹
    c2ws = []
    for i in range(frames):
        theta = 2.0 * np.pi * n_rot * (i / frames)
        pos = look_at + radius * (np.cos(theta) * forward0 + np.sin(theta) * right0)

        forward_dir = (look_at - pos)
        forward_dir /= np.linalg.norm(forward_dir)

        right_dir = np.cross(forward_dir, up)
        right_dir /= np.linalg.norm(right_dir)
        up_dir = np.cross(right_dir, forward_dir)

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = -right_dir
        c2w[:3, 1] = up_dir
        c2w[:3, 2] = forward_dir
        c2w[:3, 3] = pos
        c2ws.append(c2w)

    return c2ws


def camera_ray_sample_points(
    camera,
    scene_radius: float,
    n_points: int = 256,
    mask: Bool[Tensor, "H W"] = None,
    sampling_method: str = "inbound",
) -> Float[Tensor, "N n_points 3"]:
    fx = fov2focal(camera.FoVx, camera.image_width)
    fy = fov2focal(camera.FoVy, camera.image_height)

    # Fuck this shit transpose
    c2w = torch.inverse(camera.world_view_transform.T)
    # c2w = camera.world_view_transform
    R = c2w[:3, :3]
    T = c2w[:3, 3]

    camera_space_ij = torch.meshgrid(
        torch.arange(camera.image_width, dtype=torch.float32),
        torch.arange(camera.image_height, dtype=torch.float32),
        indexing="xy",
    )
    camera_space_ij = torch.stack(camera_space_ij, dim=-1)

    if mask is None:
        camera_space_ij = camera_space_ij.reshape(-1, 2)
    else:
        camera_space_ij = camera_space_ij[mask]

    assert camera_space_ij.ndim == 2

    camera_space_ij = (
        camera_space_ij
        - torch.tensor([[camera.image_width, camera.image_height]], dtype=torch.float32)
        / 2
    )

    view_space_xy = camera_space_ij * torch.tensor([[1 / fx, 1 / fy]])
    view_space_xyz = torch.cat(
        [view_space_xy, torch.ones_like(view_space_xy[..., 0:1])], dim=-1
    )

    view_space_directions = torch.bmm(
        R[None, ...].repeat(view_space_xyz.shape[0], 1, 1), view_space_xyz[..., None]
    )[..., 0]
    view_space_xyz = view_space_directions + T[None, ...]

    distances = None
    if sampling_method == "inbound":
        distances = torch.linspace(0, scene_radius * 2, n_points)
    elif sampling_method == "segmented":
        # linear inside scene radius, linear disparity outside, I forget the exact name of this sampling strategy
        distances_inside = torch.linspace(0, scene_radius, n_points // 2)
        distances_outside = torch.linspace(0, 1, n_points // 2)
    else:
        raise ValueError(f"Unknown sampling method {sampling_method}")

    return (
        view_space_directions[..., None, :] * distances[None, ..., None]
        + view_space_xyz[..., None, :]
    )


def project(camera: Simple_Camera, points3d):
    # TODO: should be equivalent to full_proj_transform.T
    if isinstance(points3d, list):
        points3d = torch.stack(points3d, dim=0)
    w2c = camera.world_view_transform.T
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    points3d_camera = torch.einsum("ij,bj->bi", R, points3d) + T[None, ...]
    xy = points3d_camera[..., :2] / points3d_camera[..., 2:]
    ij = (
        xy
        * torch.tensor(
            [
                fov2focal(camera.FoVx, camera.image_width),
                fov2focal(camera.FoVy, camera.image_height),
            ],
            dtype=torch.float32,
            device=xy.device,
        )
        + torch.tensor(
            [camera.image_width, camera.image_height],
            dtype=torch.float32,
            device=xy.device,
        )
        / 2
    ).to(torch.long)

    return ij


def unproject(camera: Simple_Camera, points2d, depth):
    origin = camera.camera_center
    w2c = camera.world_view_transform.T
    R = w2c[:3, :3].T

    if isinstance(points2d, (list, tuple)):
        points2d = torch.stack(points2d, dim=0)

    points2d[0] *= camera.image_width
    points2d[1] *= camera.image_height
    points2d = points2d.to(w2c.device)
    points2d = points2d.to(torch.long)

    directions = (
        points2d
        - torch.tensor(
            [camera.image_width, camera.image_height],
            dtype=torch.float32,
            device=w2c.device,
        )
        / 2
    ) / torch.tensor(
        [
            fov2focal(camera.FoVx, camera.image_width),
            fov2focal(camera.FoVy, camera.image_height),
        ],
        dtype=torch.float32,
        device=w2c.device,
    )
    padding = torch.ones_like(directions[..., :1])
    directions = torch.cat([directions, padding], dim=-1)
    if directions.ndim == 1:
        directions = directions[None, ...]
    directions = torch.einsum("ij,bj->bi", R, directions)
    directions = F.normalize(directions, dim=-1)

    points3d = (
        directions * depth[0][points2d[..., 1], points2d[..., 0]] + origin[None, ...]
    )

    return points3d


def get_point_depth(points3d, camera: Simple_Camera):
    w2c = camera.world_view_transform.T
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    points3d_camera = torch.einsum("ij,bj->bi", R, points3d) + T[None, ...]
    depth = points3d_camera[..., 2:]
    return depth
