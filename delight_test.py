import copy
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import pipeline
from gaussiansplatting.arguments import PipelineParams
from gaussiansplatting.scene.camera_scene import CamScene
from argparse import ArgumentParser
from tqdm import tqdm
import yaml
import json

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

def safe_name(s: str):
    return "".join(c if c.isalnum() or c in ('_', '-') else "_" for c in s.replace(" ", "_"))

# Loading
print("Loading RMBG model...")
rmbg_pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
# rmbg_pipe = pipeline("image-segmentation", model="/data/vjuicefs_ai_camera_ql/11184175/models/RMBG-1.4", trust_remote_code=True)
print("Loading StableNormal model...")
normal_model = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)
# normal_model = torch.hub.load("/data/juicefs_sharing_data/11184175/work/dge-iclight/StableNormal", "StableNormal_turbo", trust_repo=True, source="local", local_cache_dir="/data/vjuicefs_ai_camera_ql/11184175/models/stable-normal/weights")
print("Loading StableDelight model...")
delight_model = torch.hub.load("Stable-X/StableDelight", "StableDelight_turbo", trust_repo=True)
# delight_model = torch.hub.load("/data/juicefs_sharing_data/11184175/work/StableDelight", "StableDelight_turbo", trust_repo=True, source="local", local_cache_dir="/data/vjuicefs_ai_camera_ql/11184175/models/stable-delight/weights")


# Main
parser = ArgumentParser(description="Testing script parameters")
pipeline = PipelineParams(parser)

source_config = Path("script/benchmark/data_source.yaml")
prompt_json = Path("script/benchmark/scene_prompt_dir.json")
output_folder = Path("temp/delight_test")
os.makedirs(output_folder, exist_ok=True)

print("Loading data source config:", source_config)
with open(source_config, "r") as f:
    data_sources = yaml.safe_load(f)

print("Loading scene prompts:", prompt_json)
with open(prompt_json, "r") as f:
    scene_prompts = json.load(f)

task_base_folder = Path("/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_ablation_mv/full_dir")
metrics_save_folder = Path("evaluation_results/dge_ablation_mv/full_dir")
metrics_save_folder.mkdir(exist_ok=True)
metrics_corr = {}
total_corr = []
for dataset_name, dataset_info in data_sources.items():
    if dataset_name != "in2n": continue
    dataset_folder = task_base_folder / dataset_name
    if not dataset_folder.exists():
        print(f"  [Warning] Path {str(dataset_folder)} does not exsits")
        
    scenes = dataset_info.get("scenes", [])
    base_path = Path(dataset_info['base_path'])
    gs_base_path = Path(dataset_info['gs_base_path'])
    exp_folder = dataset_info['exp_folder'] if "exp_folder" in dataset_info else ""

    print(f"\nProcessing dataset: {dataset_name}, total scenes: {len(scenes)}")

    dataset_corrs = []
    for scene_name in tqdm(scenes):
        scene_folder = dataset_folder / scene_name
        if not scene_folder.exists():
            print(f"  [Warning] Path {str(scene_folder)} does not exsits")
            
        scene_key = f"{dataset_name}/{scene_name}"
        
        # obtain prompts
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
        
        data_source = base_path / scene_full
        exp_path = gs_base_path / scene_full / exp_folder if exp_folder else ""

        for idx in range(3):
            prompt = prompts[idx]
            # task_name = safe_name(prompt)
            # task_folder = scene_folder / task_name
            task_name = f"dge-iclight_{dataset_name}_{scene_name}_{idx}"
            task_folder = task_base_folder / task_name
            if not task_folder.exists():
                print(f"  [Warning] Path {str(task_folder)} does not exsits")
            
            # get DIR groundtruth
            dir_gt = prompt.split(' ')[-1]
            if dir_gt not in ["right", "left", "top", "bottom"]:
                print(f"[ERROR] Wrong direction parsed from prompt: {dir_gt}")
                continue

            for run_folder in task_folder.iterdir():
                # prepare images
                relit_image_folder = run_folder / "save" / "edited_images"  # relit_images是traning views的顺序，CamScene映射到70个id需要注意
                # relit_image_folder = run_folder / "relit_images"  # relit_images是traning views的顺序，CamScene映射到70个id需要注意
                img_files = sorted(relit_image_folder.glob("*.png"))
                
                # loading cameras
                config_path = run_folder / "configs" / "parsed.yaml"
                with open(config_path, "r") as f:
                    parsed_config = yaml.safe_load(f)
                ref_id = parsed_config['system']['ref_id']
                cam_scene = CamScene(str(data_source), h=512, w=512, transforms_path=exp_path, image_limit=70)
                train_views = cam_scene.cameras
                training_poses = get_poses_from_cams(train_views)  # pose是w2c
                w2c_ref = training_poses[ref_id]
                c2w_ref = np.linalg.inv(w2c_ref)
                anchor_cam = np.array([
                    [0.0, 0.0, 0.0], # origin
                    [-1.0, 0.0, 0.0], # right
                    [1.0, 0.0, 0.0], # left
                    [0.0, 1.0, 0.0], # top
                    [0.0, -1.0, 0.0] # bottom
                ])  # (5, 3)
                anchor_cam_h = np.concatenate([anchor_cam, np.ones((len(anchor_cam), 1), dtype=anchor_cam.dtype)], axis=1)  # (5, 4)
                anchor_w_h = (c2w_ref @ anchor_cam_h.T).T
                
                # TT = len(img_files)
                # RP = min(5, TT)
                # # TODO, 检查一下ref在不在img_files，在的话就包含他
                # r = np.random.choice(TT, RP, replace=False)
                # pick_ids = [ref_id] + r.tolist()
                # pick_files = [str(img_files[idx]) for idx in pick_ids]

                TT = len(img_files)
                RP = min(5, TT)
                r = np.random.choice(TT, RP, replace=False)
                pick_ids = r.tolist()
                # dirfiles = sorted(relit_image_folder.glob("*"))
                # ext = dirfiles[0].suffix
                pick_filenumbers = [int(img_files[idx].stem) for idx in pick_ids]
                if ref_id not in pick_filenumbers:
                    pick_filenumbers[0] = ref_id  # ensure ref is included
                pick_files = [str(relit_image_folder / f"{num:03d}.png") for num in pick_filenumbers]
            
                # calculate picked views
                diff_sel_list = []
                dirsim_sel_list = []
                for idx, image_path in zip(pick_ids, pick_files):
                    w2c = training_poses[idx]
                    anchor_c_h = (w2c @ anchor_w_h.T).T
                    direction_dict = {
                        "right": anchor_c_h[1, :3] - anchor_c_h[0, :3],
                        "left": anchor_c_h[2, :3] - anchor_c_h[0, :3],
                        "top": anchor_c_h[3, :3] - anchor_c_h[0, :3],
                        "bottom": anchor_c_h[4, :3] - anchor_c_h[0, :3],
                    }
                    
                    # print(f"Processing view {image_path.name}")
                    if not Path(image_path).exists():
                        print(f"  [Warning] Image {str(image_path)} does not exsits")
                        continue
                    input_image = Image.open(image_path)
                    input_gray = input_image.convert("L")
                    input_np = np.array(input_gray).astype(np.float32)

                    # print("Generating foreground mask using RMBG...")
                    front_mask = rmbg_pipe(input_image, return_mask = True) # outputs a pillow mask
                    mask_np = np.array(front_mask).astype(np.float32) / 255.0
                    mask = mask_np > 0.5

                    # print("Applying StableDelight model...")
                    delight_image = delight_model(input_image)
                    delight_gray = delight_image.convert("L")
                    delight_np = np.array(delight_gray).astype(np.float32)

                    # print("Applying StableNormal model...")
                    normal_image = normal_model(input_image)
                    normal_np = np.array(normal_image).astype(np.float32) - 127.5  # shape (512, 512, 3), range [-127.5, 127.5]
                    norm = np.linalg.norm(normal_np, axis=2, keepdims=True)   # shape (512, 512, 1)
                    normals = normal_np / (norm + 1e-8) # [-1, 1], unit vectors

                    # print("Computing average normal in bright regions...")
                    normal_masked = normal_np * mask_np[..., None]
                    diff_masked = np.clip(input_np - delight_np, a_min=0.0, a_max=255.0).squeeze() / 255.0
                    diff_sel = diff_masked[mask]
                    
                    dir_vector = direction_dict[dir_gt]
                    dir_sim = np.clip((normals * dir_vector).sum(axis=2), a_min=0.0, a_max=1.0)  # shape (512, 512)
                    dir_sim_sel = dir_sim[mask]
                    
                    diff_sel_list.append(diff_sel)
                    dirsim_sel_list.append(dir_sim_sel)


                    # for dir_name, dir_vector in direction_dict.items():
                    #     dir_sim = np.clip((normals * dir_vector).sum(axis=2), a_min=0.0, a_max=1.0)  # shape (512, 512)
                    #     dir_sim_255 = dir_sim * 255.0
                    #     dir_sim_pil = Image.fromarray(dir_sim_255.astype(np.uint8))
                    #     dir_sim_pil.save(output_folder / f"dir_sim_{dir_name}_{idx:03}.png")
                    #     dir_sim_sel = dir_sim[mask]
                    #     corr = np.corrcoef(diff_sel, dir_sim_sel)[0, 1]
                    #     print(f"{dir_name}: {corr:.4f}")
                    
                    # Save or display the result
                    # if visualize := True:
                    #     print(f"Saving visualization results to {str(output_folder)}")
                    #     front_mask.save(output_folder / f"mask_{idx:03}.png")
                    #     input_image.save(output_folder / f"input_{idx:03}.png")
                    #     delight_image.save(output_folder / f"delight_{idx:03}.png")
                    #     normal_image.save(output_folder / f"normal_{idx:03}.png")
                    #     input_gray.save(output_folder / f"input_gray_{idx:03}.png")
                    #     delight_gray.save(output_folder / f"delight_gray_{idx:03}.png")
                    #     diff_masked = diff_masked * 255.0
                    #     diff_masked_pil = Image.fromarray(diff_masked.astype(np.uint8))
                    #     diff_masked_pil.save(output_folder / f"diff_masked_{idx:03}.png")

                # calculate corr
                diffs = np.concatenate(diff_sel_list)
                dirsims = np.concatenate(dirsim_sel_list)
                corr = np.corrcoef(diffs, dirsims)[0, 1]
                print(f"{dir_gt}-DirCorr: {corr:.4f} | {task_name}")
                dataset_corrs.append(corr)
                
                # add corr to metrics.json
                # task_metrics_path = task_folder / "metrics.json"
                task_metrics_path = run_folder / "save" / "metrics.json"
                try:
                    with open(task_metrics_path, "r") as fp:
                        task_metrics = json.load(fp)
                except:
                    task_metrics = {}
                task_metrics["dir_corr"] = corr
                with open(task_metrics_path, "w") as fp:
                    json.dump(task_metrics, fp, indent=2)
    
    dataset_corrs_mean = np.array(dataset_corrs).mean()
    metrics_corr[dataset_name] = dataset_corrs_mean
    total_corr.extend(dataset_corrs)

    
metrics_corr["avg"] = np.array(total_corr).mean()
metrics_save_path = metrics_save_folder / "metrics_corr.json"
with open(metrics_save_path, "w") as fp:
    json.dump(metrics_corr, fp, indent=2)
    print(f"Dir Corr metrics has been saved to {str(metrics_save_path)}")