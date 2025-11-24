import os
import json
import mediapy
import torch
import yaml
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, rotmat2qvec, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from torchvision import transforms
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

# 加载 CLIP 模型
print(">>> 加载CLIP模型")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model = CLIPModel.from_pretrained("/mnt/16T/yejiangnan/.cache/hf_models/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("/mnt/16T/yejiangnan/.cache/hf_models/clip-vit-large-patch14")

def clip_preprocess_tensor(tensor: torch.Tensor):
    # tensor shape: (1, 3, H, W)
    if tensor.shape[1] != 3:
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.shape[-1] != 224 or tensor.shape[-2] != 224:
        tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
    # Normalize
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std

def compute_clip_metrics(img_before: torch.Tensor, img_after: torch.Tensor, text: str, neutral_text="a photo"):
    """
    img_before, img_after: Tensor of shape (1, 3, H, W), range [0,1]
    """
    img_before = clip_preprocess_tensor(img_before)
    img_after = clip_preprocess_tensor(img_after)

    # Get CLIP features
    image_emb_before = clip_model.get_image_features(img_before)
    image_emb_after = clip_model.get_image_features(img_after)

    # Text encoding
    text_inputs = clip_processor(text=[text, neutral_text], return_tensors="pt", padding=True).to(device)
    text_embs = clip_model.get_text_features(**text_inputs)
    text_emb, neutral_emb = text_embs[0], text_embs[1]

    # Normalize
    image_emb_before = F.normalize(image_emb_before, dim=-1)
    image_emb_after = F.normalize(image_emb_after, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    neutral_emb = F.normalize(neutral_emb, dim=-1)

    # CLIP similarity
    clip_sim = torch.cosine_similarity(image_emb_after, text_emb, dim=-1).item()

    # Directional similarity
    image_dir = image_emb_after - image_emb_before
    text_dir = text_emb - neutral_emb
    dir_sim = F.cosine_similarity(image_dir, text_dir, dim=-1).item()

    return clip_sim, dir_sim


class CenterCropSquareResize:
    def __init__(self, size=512):
        self.size = size

    def __call__(self, img: Image.Image):
        w, h = img.size
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        img = img.crop((left, top, right, bottom))  # 中心裁剪正方形
        img = img.resize((self.size, self.size), Image.Resampling.LANCZOS)
        return img

MY_transform = transforms.Compose([
    CenterCropSquareResize(512),
    np.array
])

MY_transform_ts = transforms.Compose([
    CenterCropSquareResize(512),
    transforms.ToTensor(),
])

def safe_name(s: str):
    return "".join(c if c.isalnum() or c in ('_', '-') else "_" for c in s.replace(" ", "_"))

def load_center_crop_resize(image_path: str, size: int = 512):
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    h, w, _ = img_bgr.shape
    crop_size = min(h, w)
    y1 = (h - crop_size) // 2
    x1 = (w - crop_size) // 2
    img_cropped = img_bgr[y1:y1 + crop_size, x1:x1 + crop_size]
    img_resized = cv2.resize(img_cropped, (size, size), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return img_rgb  # uint8, 0~255, shape=(H,W,3)

def get_sorted_idx(c2ws):
    foward_vectos = [c2w[:3, 2] for c2w in c2ws]
    foward_vectos = np.array(foward_vectos)
    cams_center_x = np.array([c2w[0, 3].item() for c2w in c2ws])
    most_left_vecotr = foward_vectos[np.argmin(cams_center_x)]
    distances = [np.arccos(np.clip(np.dot(most_left_vecotr, c2w[:3, 2]), 0, 1)) for c2w in c2ws]
    sorted_c2ws = [c2w for _, c2w in sorted(zip(distances, c2ws), key=lambda pair: pair[0])]
    reference_axis = np.cross(most_left_vecotr, sorted_c2ws[1][:3, 2])
    distances_with_sign = [np.arccos(np.dot(most_left_vecotr, c2w[:3, 2])) if np.dot(reference_axis,  np.cross(most_left_vecotr, c2w[:3, 2])) >= 0 else 2 * np.pi - np.arccos(np.dot(most_left_vecotr, c2w[:3, 2])) for c2w in c2ws]
    sorted_cam_idx = [idx for _, idx in sorted(zip(distances_with_sign, range(len(c2ws))), key=lambda pair: pair[0])]
    return sorted_cam_idx

def get_c2ws_filepaths(dataset_name: str, full_base):
    if dataset_name in ['in2n', 'scannetpp']:  # load from json
        if dataset_name == 'in2n':
            json_path = Path(full_base) / "transforms.json"
            image_subfolder = "images"
        elif dataset_name == 'scannetpp':
            json_path = Path(full_base) / "nerfstudio/transforms_undistorted.json"
            image_subfolder = "resized_undistorted_images"
        with open(json_path, "r") as fp:
            transforms = json.load(fp)
            
        c2ws = [np.array(frame["transform_matrix"]) for frame in transforms["frames"]]
        filenames = [Path(frame["file_path"]).name for frame in transforms["frames"]]
        filepaths = [Path(full_base) / image_subfolder / fn for fn in filenames]
        return c2ws, filepaths

    elif dataset_name in ['mip360']:  # load from colmap
        image_subfolder = "images"
        sparse_path = Path(full_base) / "sparse/0"
        images_bin = sparse_path / "images.bin"
        cam_extrinsics = read_extrinsics_binary(images_bin)
        c2ws = []
        filepaths = []
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            w2c_R = qvec2rotmat(extr.qvec)
            w2c_T = np.array(extr.tvec)
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = w2c_R
            w2c[:3, 3] = w2c_T
            c2w = np.linalg.inv(w2c)
            filename = Path(extr.name).name
            filepaths.append(Path(full_base) / image_subfolder / filename)
            c2ws.append(c2w)
        return c2ws, filepaths
    else:
        raise RuntimeError(f"Unsupported scene name: {dataset_name}")
    
def prepare_images_from_filepaths(filepaths):
    # return (F,H,W,C)
    img_tensors = np.stack([MY_transform(Image.open(fp)) for fp in filepaths])
    return img_tensors

def main(source_config, prompt_json, output_root, overwrite=False, max_imgs=17):
    print("Loading data source config:", source_config)
    with open(source_config, "r") as f:
        data_sources = yaml.safe_load(f)

    print("Loading scene prompts:", prompt_json)
    with open(prompt_json, "r") as f:
        scene_prompts = json.load(f)

    output_root = Path(output_root)
    # output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_metrics = {"avg": {"clip_sim": [], "clip_dir": []}}
    for dataset_name, dataset_info in data_sources.items():
        if dataset_name != 'in2n': continue
        scenes = dataset_info.get("scenes", [])
        base_path = Path(dataset_info['base_path'])
        print(f"\nProcessing dataset: {dataset_name}, total scenes: {len(scenes)}")
        dataset_metrics = {"clip_sim": [], "clip_dir": []}
        for scene_name in scenes:
            scene_key = f"{dataset_name}/{scene_name}"
            if scene_key not in scene_prompts:
                print(f"  [Warning] No prompts for {scene_key}, skip.")
                continue
            prompts = scene_prompts[scene_key]
            if not prompts:
                print(f"  [Warning] Empty prompts for {scene_key}, skip.")
                continue

            scene_subfolder = dataset_info.get("scene_subfolder", None)
            if scene_subfolder:
                scene_full = f"{scene_name}/{scene_subfolder}"
            else:
                scene_full = scene_name
            # image_subfolder = dataset_info.get("image_subfolder", "images")
            # imgs_folder = base_path / scene_full / image_subfolder
            
            c2ws, _ = get_c2ws_filepaths(dataset_name, base_path/scene_full)
            image_limit = 70
            train_num = len(c2ws)
            if train_num > image_limit:
                print(f"[INFO] Too many cameras, randomly select {image_limit} cameras for training and testing.")
                rng = np.random.default_rng(seed=42)
                train_num = min(train_num, image_limit)
                train_ids = rng.permutation(train_num)[:image_limit]
                c2ws = [c2ws[i] for i in train_ids]
                # img_paths = [img_paths[i] for i in train_ids]
                
            # sorted_idx = get_sorted_idx(c2ws)

            for idx in range(3):
                prompt = prompts[idx]
                task_folder = output_root / f"dge-iclight_{dataset_name}_{scene_name}_{idx}"
                if not task_folder.exists():
                    print(f"  [Warning] Task folder does not exist: {task_folder}, skip.")
                    continue
                metrics = {"clip_sim": [], "clip_dir": []}
                for run_folder in task_folder.iterdir():
                    save_folder = run_folder / "save"
                    origin_img_folder = save_folder / "origin_images"
                    relit_img_folder = save_folder / "edited_images"
                    
                    origin_img_paths = list(origin_img_folder.glob("*.png"))
                    relit_img_paths = list(relit_img_folder.glob("*.png"))
                    # select_ids = [int(img_path.stem) for img_path in origin_img_paths]
                    # c2ws_selected = [c2ws[i] for i in select_ids]
                    # sorted_idx = get_sorted_idx(c2ws_selected)

                    # sorted_orig_img_paths = [origin_img_paths[i] for i in sorted_idx]
                    # sorted_relit_img_paths = [relit_img_paths[i] for i in sorted_idx]

                    for ori_img_path, relit_img_path in zip(origin_img_paths, relit_img_paths):
                        img_before = MY_transform_ts(Image.open(ori_img_path)).unsqueeze(0).to(device)
                        img_after = MY_transform_ts(Image.open(relit_img_path)).unsqueeze(0).to(device)
                    
                        clip_sim, clip_dir = compute_clip_metrics(img_before, img_after, prompt)
                        metrics["clip_sim"].append(clip_sim)
                        metrics["clip_dir"].append(clip_dir)
                
                metrics["clip_sim"] = np.mean(np.array(metrics["clip_sim"])).item()
                metrics["clip_dir"] = np.mean(np.array(metrics["clip_dir"])).item()
                image_metrics_path = save_folder / "image_metrics.json"
                with open(image_metrics_path, "w") as fp:
                    json.dump(metrics, fp, indent=2)
                print(f"  Saved image metrics to {image_metrics_path}")
                dataset_metrics["clip_sim"].append(metrics["clip_sim"])
                dataset_metrics["clip_dir"].append(metrics["clip_dir"])
                        
                
                relit_video_path = save_folder / "relit_images.mp4"
                if relit_video_path.exists() and not overwrite:
                    print(f"  [Skip] Video already exists: {relit_video_path}")
                    continue

                # # Load all images in the same sorted order
                # relit_images = []
                # for relit_img_path in sorted_relit_img_paths:
                #     img = np.array(Image.open(relit_img_path))
                #     relit_images.append(img)
                   

                # if len(relit_images) > 0:
                #     mediapy.write_video(relit_video_path, relit_images, fps=10)
                #     print(f"  Saved video to {relit_video_path}")
                # else:
                #     print(f"  [Warning] No relit images found for {str(save_folder)}")

        full_metrics["avg"]["clip_sim"].extend(dataset_metrics["clip_sim"])
        full_metrics["avg"]["clip_dir"].extend(dataset_metrics["clip_dir"])
        dataset_metrics["clip_sim"] = np.mean(np.array(dataset_metrics["clip_sim"])).item()
        dataset_metrics["clip_dir"] = np.mean(np.array(dataset_metrics["clip_dir"])).item()
        full_metrics[dataset_name] = dataset_metrics

        print(f"\nDataset {dataset_name} metrics:")
    
    full_metrics["avg"]["clip_sim"] = np.mean(np.array(full_metrics["avg"]["clip_sim"])).item()
    full_metrics["avg"]["clip_dir"] = np.mean(np.array(full_metrics["avg"]["clip_dir"])).item()
    metrics_json_path = Path("full_metrics.json") 
    with open(metrics_json_path, "w") as fp:
        json.dump(full_metrics, fp, indent=2)
    print(f"\nSaved full metrics to {metrics_json_path}")
    print("\nAll relighting completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relight scenes with generated backgrounds")
    parser.add_argument("--source-config", type=str, default="script/benchmark/data_source.yaml")
    parser.add_argument("--prompt-json", type=str, default="script/benchmark/scene_prompt_dir.json")
    parser.add_argument("--output-root", type=str, default="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_ablation_mv/full_dir",
                        help="Output root folder for relit images")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing images")
    parser.add_argument("--max-imgs", type=int, default=17, help="Max images per scene to process")
    args = parser.parse_args()

    main(args.source_config, args.prompt_json, args.output_root,
         overwrite=args.overwrite, max_imgs=args.max_imgs)