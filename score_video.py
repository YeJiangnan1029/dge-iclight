import os
from pathlib import Path
from tqdm import tqdm
from vbench import VBench
import yaml

source_config = Path("script/benchmark/data_source.yaml")
output_folder = Path("temp/trajectory_vis")
os.makedirs(output_folder, exist_ok=True)

print("Loading data source config:", source_config)
with open(source_config, "r") as f:
    data_sources = yaml.safe_load(f)

task_base_folder = Path("/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_ablation_mv/full_dir")
print(f"Task base folder: {task_base_folder}")

video_paths = []
for dataset_name, dataset_info in data_sources.items():
    if dataset_name != 'in2n': continue
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
            task_name = f"dge-iclight_{dataset_name}_{scene_name}_{idx}"
            task_folder = task_base_folder / task_name
            for exp_subfolder in task_folder.iterdir():
                if not exp_subfolder.is_dir():
                    continue
                print(f"Rendering scene: {scene_key}, experiment: {exp_subfolder.name}")

                video_path = str(exp_subfolder / "save" / "edited_scene.mp4")
                video_paths.append(video_path)

print(f"Collected {len(video_paths)} videos for scoring.")

my_VBench = VBench("cuda", "/mnt/16T/yejiangnan/work/VBench/vbench/VBench_full_info.json", "evaluation_results")
my_VBench.evaluate(
    videos_path = video_paths,
    name = "trajectory",
    dimension_list = ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality"],
    mode = "custom_input"
)