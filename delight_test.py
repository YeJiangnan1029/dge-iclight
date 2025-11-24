import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import pipeline

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print("Loading RMBG model...")
rmbg_pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
print("Loading StableNormal model...")
normal_model = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)
print("Loading StableDelight model...")
delight_model = torch.hub.load("Stable-X/StableDelight", "StableDelight_turbo", trust_repo=True)

image_path = Path("/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir/dge-iclight_in2n_bear_1/bear,_woodland,_twilight_glow,_golden_hour,_soft_light,_light_from_the_left@20251104-032748/save/edited_images/000.png")
print(f"Loading image {image_path.name}")
input_image = Image.open(image_path)
input_gray = input_image.convert("L")
input_np = np.array(input_gray).astype(np.float32)

print("Generating foreground mask using RMBG...")
front_mask = rmbg_pipe(input_image, return_mask = True) # outputs a pillow mask
mask_np = np.array(front_mask).astype(np.float32) / 255.0
mask = mask_np > 0.5

print("Applying StableDelight model...")
delight_image = delight_model(input_image)
delight_gray = delight_image.convert("L")
delight_np = np.array(delight_gray).astype(np.float32)

print("Applying StableNormal model...")
normal_image = normal_model(input_image)
normal_np = np.array(normal_image).astype(np.float32) - 127.5  # shape (512, 512, 3), range [-127.5, 127.5]
norm = np.linalg.norm(normal_np, axis=2, keepdims=True)   # shape (512, 512, 1)
normals = normal_np / (norm + 1e-8) # [-1, 1], unit vectors

print("Computing average normal in bright regions...")
direction_dict = {
    "right": np.array([-1.0, 0.0, 0.0]),
    "left": np.array([1.0, 0.0, 0.0]),
    "top": np.array([0.0, 1.0, 0.0]),
    "bottom": np.array([0.0, -1.0, 0.0]),
}
normal_masked = normal_np * mask_np[..., None]
diff_masked = np.clip(input_np - delight_np, a_min=0.0, a_max=255.0).squeeze() / 255.0
diff_sel = diff_masked[mask]
for dir_name, dir_vector in direction_dict.items():
    dir_sim = np.clip((normals * dir_vector).sum(axis=2), a_min=0.0, a_max=1.0)  # shape (512, 512)
    dir_sim_sel = dir_sim[mask]
    corr = np.corrcoef(diff_sel, dir_sim_sel)[0, 1]
    print(f"{dir_name}: {corr:.4f}")
    # dir_sim_pil = Image.fromarray((dir_sim * 255.0).astype(np.uint8))
    # dir_sim_pil.save(Path("temp/delight_test") / f"normal_dirsim_{dir_name}.png")
    

# Save or display the result
if visualize := True:
    vis_path = Path("temp/delight_test")
    vis_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualization results to {str(vis_path)}")
    front_mask.save(vis_path / "mask.png")
    input_image.save(vis_path / "input.png")
    delight_image.save(vis_path / "delight.png")
    normal_image.save(vis_path / "normal.png")
    input_gray.save(vis_path / "input_gray.png")
    delight_gray.save(vis_path / "delight_gray.png")
    normal_masked_pil = Image.fromarray(normal_masked.astype(np.uint8))
    normal_masked_pil.save(vis_path / "normal_masked.png")
    diff_masked = diff_masked * 255.0
    diff_masked_pil = Image.fromarray(diff_masked.astype(np.uint8))
    diff_masked_pil.save(vis_path / "diff_masked.png")

