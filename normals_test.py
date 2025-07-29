import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import pathlib
from PIL import Image
from torchvision.transforms import transforms as TF
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kornia.geometry.depth import depth_to_normals
from kornia.color import ColorMap, ColorMapType
from kornia.color import normals_to_rgb255, ApplyColorMap, apply_colormap, rgb_to_rgb255
import kornia.filters as KF

def get_camera_centers(w2cs):
    # 从 [R|t] 中求相机中心 C = -R^T t
    R = w2cs[:, :3, :3]
    t = w2cs[:, :3, 3:]
    centers = -torch.bmm(R.transpose(1, 2), t).squeeze(-1)  # (N, 3)
    return centers

def compute_similarity_transform(X, Y):
    """
    Estimate Sim(3) transformation: Y ≈ s * R @ X + t
    X: (N, 3), predicted centers (torch.Tensor)
    Y: (N, 3), GT centers (torch.Tensor)
    Returns:
        scale: float
        R: (3, 3) torch.Tensor
        t: (3,) torch.Tensor
    """
    assert X.shape == Y.shape and X.shape[1] == 3, "Input shape must be (N, 3)"
    
    X_mean = X.mean(dim=0)  # (3,)
    Y_mean = Y.mean(dim=0)  # (3,)
    Xc = X - X_mean         # (N, 3)
    Yc = Y - Y_mean         # (N, 3)

    # Rotation (SVD of Xc^T Yc)
    H = Xc.T @ Yc  # (3, 3)
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Scale
    var_X = (Xc ** 2).sum()
    scale = (S.sum()) / var_X

    # Translation
    t = Y_mean - scale * R @ X_mean

    return scale.item(), R, t

def apply_sim_transform(w2cs, scale, R, t):
    """
    Apply similarity transformation to the camera extrinsics.
    w2cs: (N, 4, 4), original camera extrinsics
    scale: float, scaling factor
    R: (3, 3), rotation matrix
    t: (3,), translation vector
    Return: transformed w2cs
    """
    c2ws = torch.inverse(w2cs)  # (N, 4, 4)
    c2ws_R, c2ws_t = c2ws[:, :3, :3], c2ws[:, :3, 3:4]  # (N, 3, 3), (N, 3, 1)
    R = R.expand(c2ws.shape[0], -1, -1)  # Expand R to match batch size
    t = t.expand(c2ws.shape[0], -1).unsqueeze(-1)  # Expand t
    newR = torch.bmm(R, c2ws_R)  # Apply rotation
    newt = torch.bmm(R, c2ws_t) * scale + t  # Apply scaling and translation
    newc2ws = torch.eye(4, device=w2cs.device).unsqueeze(0).expand(c2ws.shape[0], -1, -1)  # Identity matrix
    newc2ws = newc2ws.contiguous()
    newc2ws[:, :3, :3] = newR
    newc2ws[:, :3, 3:4] = newt
    return torch.inverse(newc2ws)  # Return w2cs

def visualize_w2c_cameras(w2cs: torch.Tensor, ax=None, scale=0.1, color='b', label_prefix='cam'):
    """
    可视化一组 w2c 相机外参 (B, 4, 4)
    每个相机显示其坐标轴和相机中心。

    参数:
        w2cs: (B, 4, 4) tensor，相机外参（world to camera）
        ax: 可选 matplotlib 3D axes 对象
        scale: 坐标轴长度缩放
        color: 默认坐标轴颜色
    """
    B = w2cs.shape[0]
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    for i in range(B):
        w2c = w2cs[i]  # (4,4)
        R = w2c[:3, :3]  # (3,3)
        t = w2c[:3, 3]   # (3,)
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t

        cam_center = c2w[:3, 3]

        # 绘制相机坐标轴
        x_axis = cam_center + c2w[:3, 0] * scale
        y_axis = cam_center + c2w[:3, 1] * scale
        z_axis = cam_center + c2w[:3, 2] * scale

        ax.plot([cam_center[0], x_axis[0]], [cam_center[1], x_axis[1]], [cam_center[2], x_axis[2]], color='r')
        ax.plot([cam_center[0], y_axis[0]], [cam_center[1], y_axis[1]], [cam_center[2], y_axis[2]], color='g')
        ax.plot([cam_center[0], z_axis[0]], [cam_center[1], z_axis[1]], [cam_center[2], z_axis[2]], color='b')

        ax.scatter(cam_center[0], cam_center[1], cam_center[2], color=color, s=10)
        ax.text(cam_center[0], cam_center[1], cam_center[2], f"{label_prefix}{i}", fontsize=8)

    ax.set_box_aspect([1, 1, 1])
    plt.savefig(f"temp/cameras_{label_prefix}.png")
    

cams = torch.load("temp/cams.pkl")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
# model = VGGT.from_pretrained("/data/vjuicefs_ai_camera_vgroup_ql/11184175/models/VGGT-1B").to(device)
# image_names = [pathlib.Path(f"edit_cache/-data-vjuicefs_ai_camera_vgroup_ql-11184175-data-dge-face-scene_point_cloud.ply/origin_render/{cam.uid:04d}.png") for cam in cams]
image_names = [pathlib.Path(f"edit_cache/data-dge_data-face-scene_point_cloud.ply/origin_render/{cam.uid:04d}.png") for cam in cams]
n_images = 5
idx_list = list(range(0, len(image_names), len(image_names) // n_images))
image_names = [image_names[i] for i in idx_list]
cams = [cams[i] for i in idx_list]
w2cs = torch.stack([cam.world_view_transform.T for cam in cams], dim=0).to(device)
image_origin = [Image.open(path) for path in image_names]
W, H = image_origin[0].size[0], image_origin[0].size[1]
padded_W = (W + 13) // 14 * 14
padded_H = (H + 13) // 14 * 14

preprocess = TF.Compose([
    TF.Resize((padded_H, padded_W), interpolation=TF.InterpolationMode.BICUBIC),
    TF.ToTensor()
    ])
images = torch.stack([preprocess(image).to(device) for image in image_origin], dim=0)

with torch.no_grad():
    # with torch.cuda.amp.autocast(dtype=dtype):
    #     predictions = model(images)
    #     torch.save(predictions, "temp/predictions.pkl")
    
    # normal prediction
    predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)
    normal_list = []
    for idx, img in enumerate(image_origin):
        img = img.resize((padded_W, padded_H), Image.BICUBIC)
        normals_pil = predictor(img)
        normals = torch.tensor(np.array(normals_pil)).to(device).permute(2, 0, 1)
        normals = normals.float() / 255.0
        normal_list.append(normals)
        normals_pil.save(f"temp/stable_normal/normal_{idx:02d}.png")
        print(f"Saved normals to temp/stable_normal/normal_{idx:02d}.png")
    normal_map = torch.stack(normal_list, dim=0)  # (N, H, W, 3)
    
    predictions = torch.load("temp/predictions.pkl")
    w2cs_est, intri_est = pose_encoding_to_extri_intri(predictions['pose_enc'], (padded_H, padded_W))
    w2cs_est = w2cs_est.squeeze() # (S, 3, 4)
    bottom = torch.tensor([0, 0, 0, 1], dtype=w2cs_est.dtype, device=w2cs_est.device).view(1, 1, 4).expand(w2cs_est.shape[0], -1, -1)  # (B, 1, 4)
    w2cs_est = torch.cat([w2cs_est, bottom], dim=1)  # (B, 4, 4)
    w2cs = w2cs.squeeze() # (S, 4, 4)
    
    gt_centers = get_camera_centers(w2cs)
    est_centers = get_camera_centers(w2cs_est)
    
    scale, R, t = compute_similarity_transform(est_centers, gt_centers)

    w2cs_aligned = apply_sim_transform(w2cs_est, scale, R, t)
    
    aligned_centers = get_camera_centers(w2cs_aligned)
    error = torch.norm(aligned_centers - gt_centers, dim=1)
    error_before = torch.norm(est_centers - gt_centers, dim=1)
    print("对齐前误差：", error_before.mean())
    print("对齐后误差：", error.mean())
    visualize_w2c_cameras(w2cs.detach().cpu().numpy(), color='g', label_prefix='GT')
    visualize_w2c_cameras(w2cs_est.detach().cpu().numpy(), color='r', label_prefix='EST')
    visualize_w2c_cameras(w2cs_aligned.detach().cpu().numpy(), color='b', label_prefix='ALIGNED')

    depth_map = predictions['depth'].squeeze().unsqueeze(1)  # (B, 1, H, W)
    intri_est = intri_est.squeeze(0)  # (B, 3, 3)
    # points_map = predictions['world_points']  # (B, H, W, 3)
    depth_map = depth_map * scale
    depth_map = KF.gaussian_blur2d(depth_map, kernel_size=(5, 5), sigma=(1.0, 1.0))  # (B, 1, H, W)
    # normal_map = depth_to_normals(depth_map, intri_est)

    # image_cmp = torch.cat([image, shading_rgb], dim=2)  # (3, H, 2W)
    # image_cmp_pil = to_pil_image(image_cmp)
    # image_cmp_pil.save(f"temp/vis_shading_smoothed/shading_{i:02d}.png")
    # print(f"Saved shading visualization for image {i:02d}")

    # normal visualization
    depth_map_normalize = (depth_map-depth_map.min()) / depth_map.max()  # Normalize depth map to [0, 1]
    mapping = ApplyColorMap(ColorMap(base=ColorMapType.jet, num_colors=4096, device=depth_map.device))
    depth_vis = rgb_to_rgb255(mapping(depth_map_normalize))  # Convert to RGB for visualization
    normal_vis = normals_to_rgb255(normal_map)
    for i, (image, depth, normal) in enumerate(zip(images, depth_vis, normal_vis)):
        image = image * 255.0
        image_cmp = torch.cat([image, depth, normal], dim=2).to(torch.uint8)  # (3, H, W) + (3, H, W) = (3, H, 2W)
        image_cmp = to_pil_image(image_cmp.cpu())
        image_cmp.save(f"temp/vis_normal_smoothed/normal_{i:02d}.png")
        print(f"Saved normal visualization for image {i:02d}")
    
    # we get normal map to use
    # 第0张图的光源位置：相机中心（在世界坐标系下）
    light_pos_world = get_camera_centers(w2cs)[0]  # (3,) tensor
    n_frames = 30
    dxyz_camera0 = torch.tensor([5.0, 5.0, 0], device=light_pos_world.device)  # 光源位置的微小变化
    c2w0 = torch.inverse(w2cs[0])  # (4, 4)
    
    for j in range(n_frames):
        # 在第0张图的光源位置基础上，添加一个小的随机扰动
        light_pos_world =  c2w0[:3, :3] @ dxyz_camera0 * ((j*2-n_frames) / n_frames) + c2w0[:3, 3]
        num_views = 4
        frames = []
        for i in range(min(num_views, len(images))):
            # 当前相机的 w2c 矩阵
            w2c = w2cs[i]  # (4, 4)

            # 将 light_pos_world 转为当前视角下的相机坐标系
            light_pos_cam = w2c[:3, :3] @ light_pos_world + w2c[:3, 3]  # (3,)

            # 获取当前的 depth map 和 normal map
            depth = depth_map[i]  # (1, H, W)
            normals = normal_map[i]  # (3, H, W)

            # 构建像素网格并反投影到相机坐标系下的 3D 点（使用内参）
            _, H, W = depth.shape
            y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing='ij')
            xy_homo = torch.stack([x, y, torch.ones_like(x)], dim=0).float()  # (3, H, W)

            intri_inv = torch.inverse(intri_est[i])  # (3, 3)
            cam_rays = intri_inv @ xy_homo.view(3, -1)  # (3, H*W)
            cam_points = cam_rays * depth.view(1, -1)  # (3, H*W)

            # l: 光源方向向量 = light_pos_cam - 每个像素的3D点
            l = light_pos_cam.view(3, 1) - cam_points  # (3, H*W)
            l = l / (torch.norm(l, dim=0, keepdim=True) + 1e-6)  # 单位化 (3, H*W)

            n = normals.view(3, -1)  # (3, H*W)
            diffuse = (n * -l).sum(dim=0).clamp(min=0.0)  # (H*W,)

            # 变换回图像 (H, W)，转为灰度图显示
            shading = diffuse.view(H, W)  # (H, W)
            shading_vis = (shading * 255.0).clamp(0, 255).to(torch.uint8).cpu()

            # 可视化合成：原图 + 光照图
            image = (images[i] * 255).to(torch.uint8).cpu()  # (3, H, W)
            shading_rgb = torch.stack([shading_vis]*3, dim=0)  # (3, H, W)
            frames.append(shading_rgb)

        frames = torch.cat(frames, dim=2)
        image_cmp_pil = to_pil_image(frames.cpu())
        image_cmp_pil.save(f"temp/vis_shading_anime/frame_{j:03d}.png")
        print(f"Saved frame {j:02d}")