import gc
from typing import Dict, List
import torch
from torchvision.transforms import transforms as TF
from torch.nn.functional import interpolate
from torchvision.transforms.functional import to_pil_image

from gaussiansplatting.scene.cameras import Simple_Camera
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import kornia.filters as KF

# We use VGGT to generate depth and StableNormal to normals. 
class GeometryModel:
    
    def __init__(self, vggt_path: str|None=None, stable_normal_path: str|None=None, device=None) -> None:
        if vggt_path is None:
            vggt_path = "facebook/VGGT-1B"
            # vggt_path = "/data/vjuicefs_ai_camera_vgroup_ql/11184175/models/VGGT-1B"
        self.vggt_path = vggt_path
        
        if stable_normal_path is None:
            stablenormal_path = "facebook/VGGT-1B"
            # stable_normal_path = "/data/vjuicefs_ai_camera_vgroup_ql/11184175/models/stable-normal/weights"
        self.stable_normal_path = stable_normal_path
            
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
            
        self.vggt = None
        self.stable_normal = None
        

    def load_vggt(self):
        if self.vggt is None:
            print("Loading VGGT model...")
            self.vggt = VGGT.from_pretrained(
                self.vggt_path
            ).to(self.device)
            
    def load_stable_normal(self):
        if self.stable_normal is None:
            print("Loading StableNormal model...")
            self.stable_normal = torch.hub.load(
                "Stable-X/StableNormal", 
                "StableNormal_turbo", 
                trust_repo=True, 
                local_cache_dir=self.stable_normal_path
            )
            
    def unload_vggt(self):
        if self.vggt is not None:
            print("Unloading VGGT model...")
            del self.vggt
            self.vggt = None
            gc.collect()
            torch.cuda.empty_cache()
    
    def unload_stable_normal(self):
        if self.stable_normal is not None:
            print("Unloading StableNormal model...")
            del self.stable_normal
            self.stable_normal = None
            gc.collect()
            torch.cuda.empty_cache()

    # 进入 with 时懒加载模型
    def __enter__(self):
        # do nothing
        return self
    
    # 退出 with 时自动卸载模型并清空显存
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_vggt()
        self.unload_stable_normal()
        return False  # 如果 with 块中抛出异常，不会被吞掉
    
    def get_geometry(self, images: Dict[int, torch.Tensor], cameras: List[Simple_Camera], batch_size: int = 13):
        """
            We take input rendered images and cameras to estimate the depth and surface normals
            args:
                images: N int->torch.Tensor[float32], [0,1], (1, H, W, 3).
                cameras: List[Simple_Camera], N cameras coresponding with images.
                batch_size: how many images to process at once for VGGT to save VRAM.
            returns:
                depths: torch.Tensor[float32], (N, 1, H, W)
                normal: torch.Tensor[float32], [-1, 1](N, 3, H, W)
        """
        
        # preprocess images
        _, H, W, C = images[0].shape
        padded_W = (W + 13) // 14 * 14
        padded_H = (H + 13) // 14 * 14

        image_list = []
        camera_list = []
        for _id in range(len(cameras)):
            camera = cameras[_id]
            image = images[_id]  # (1,H,W,3)
            image = image.permute(0, 3, 1, 2)  # (1,3,H,W)
            image = interpolate(image, (padded_H, padded_W), mode="bilinear").squeeze()  # (3,H',W')
            image_list.append(image)
            camera_list.append(camera)
        image_input = torch.stack(image_list, dim=0)  # (N,3,H',W')

        # vggt inference for depth map (batched)
        self.load_vggt()
        depths_list = []
        pose_enc_list = []

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for i in range(0, len(image_input), batch_size):
                    batch = image_input[i:min(i+batch_size, len(image_input))].to(self.device, non_blocking=True)
                    vggt_predictions = self.vggt(batch)  # type: ignore
                    depths_list.append(vggt_predictions['depth'].cpu().squeeze(0))
                    pose_enc_list.append(vggt_predictions['pose_enc'].cpu().squeeze(0))

        depths_all = torch.cat(depths_list, dim=0)
        pose_enc_all = torch.cat(pose_enc_list, dim=0)

        w2cs_est, intri_est = pose_encoding_to_extri_intri(pose_enc_all.unsqueeze(0).to(self.device), (padded_H, padded_W))
        w2c_gt = torch.stack([cam.world_view_transform.T for cam in camera_list], dim=0).to(self.device)
        scale = GeometryModel.scale_align(w2c_gt, w2cs_est)

        depths = depths_all.squeeze().unsqueeze(1).to(self.device)  # (B, 1, H, W)
        depths = depths * scale
        depths = KF.gaussian_blur2d(depths, kernel_size=(5, 5), sigma=(1.0, 1.0))

        del vggt_predictions, depths_list, pose_enc_list
        self.unload_vggt()

        # stable_normal inference for normals
        image_pils = [to_pil_image(image) for image in image_list]
        normal_postprocess = TF.Compose([
            TF.ToTensor(),
            lambda x: x*2-1,  # map to [-1, 1]
        ])
        self.load_stable_normal()
        normals = []
        with torch.no_grad():
            for img_pil in image_pils:
                normal_pil = self.stable_normal(img_pil) # type: ignore
                normals.append(normal_postprocess(normal_pil))
        normals = torch.stack(normals, dim=0)
        norms = torch.sqrt((normals**2).sum(dim=1, keepdim=True).clamp_min(1e-8))
        normals = normals / norms

        return depths, normals
    
    @staticmethod
    def scale_align(w2c_gt: torch.Tensor, w2cs_est: torch.Tensor):
        w2cs_est = w2cs_est.squeeze() # (S, 3, 4)
        bottom = torch.tensor([0, 0, 0, 1], dtype=w2cs_est.dtype, device=w2cs_est.device).view(1, 1, 4).expand(w2cs_est.shape[0], -1, -1)  # (B, 1, 4)
        w2cs_est = torch.cat([w2cs_est, bottom], dim=1)  # (B, 4, 4)
        w2c_gt = w2c_gt.squeeze() # (S, 4, 4)
        
        gt_centers = GeometryModel.get_camera_centers(w2c_gt)
        est_centers = GeometryModel.get_camera_centers(w2cs_est)
        
        scale, R, t = GeometryModel.compute_similarity_transform(est_centers, gt_centers)
        # for visual
        # w2cs_aligned = GeometryModel.apply_sim_transform(w2cs_est, scale, R, t)
        return scale
        
    @staticmethod
    def get_camera_centers(w2cs):
        # 从 [R|t] 中求相机中心 C = -R^T t
        R = w2cs[:, :3, :3]
        t = w2cs[:, :3, 3:]
        centers = -torch.bmm(R.transpose(1, 2), t).squeeze(-1)  # (N, 3)
        return centers
    
    @staticmethod
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
    
    @staticmethod
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