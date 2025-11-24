import json
from pathlib import Path
import numpy as np

exp_base_folder = Path("/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark")
exp_name = "dge_ablation_mv/full_dir"
exp_folder = exp_base_folder / exp_name
output_base_folder = Path("evaluation_results")
output_folder = output_base_folder / exp_name
output_folder.mkdir(parents=True, exist_ok=True)
output_path = output_folder / "images_metrics.json"


# dataset_list = ["in2n", "mip360", "scannetpp"]
dataset_list = ["in2n"]
avg_dict = {}
total = {}
for dataset_name in dataset_list:
    tmp_dict = {}
    for task_folder in exp_folder.iterdir():
        if not task_folder.is_dir() or dataset_name not in task_folder.name:
            continue
        for run_folder in task_folder.iterdir():
            if not run_folder.is_dir():
                continue

            metrics_path = run_folder / "save" / "image_metrics.json"
            with open(metrics_path, "r") as fp:
                metrics = json.load(fp)
            
            for item in metrics:
                if item not in tmp_dict:
                    tmp_dict[item] = []
                tmp_dict[item].append(metrics[item])

                if item not in avg_dict:
                    avg_dict[item] = []
                avg_dict[item].append(metrics[item])

    tmp_metrics = {}
    for item in tmp_dict:
        value = np.mean(np.array(tmp_dict[item])).item()
        tmp_metrics[item] = value
        
    total[dataset_name] = tmp_metrics
    
for item in avg_dict:
    value = np.mean(np.array(avg_dict[item])).item()
    if "avg" not in total:
        total["avg"] = {}
    total["avg"][item] = value

with open(output_path, "w") as fp:
    json.dump(total, fp, indent=2)