from pathlib import Path
import json
import numpy as np

base_dir = Path("evaluation_results")
exp_name = "dge_ablation_mv/full_dir"
vbench_json = "evaluation_results/trajectory_eval_results.json"
metrics_json = base_dir / exp_name / "images_metrics.json"
output_json = base_dir / exp_name / "full_metrics.json"
with open(metrics_json, "r") as fp:
    metrics = json.load(fp)

# dataset_list = ["in2n", "mip360", "scannetpp"]
dataset_list = ["in2n"]
item_list = ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality"]
with open(vbench_json, "r") as fp:
    vbench_result = json.load(fp)
    
    for item in item_list:
        total = {"avg": []}
        for dataset_name in dataset_list:
            total[dataset_name] = []
        result_list = vbench_result[item][1]
        for res in result_list:
            value = res["video_results"]
            for dataset_name in dataset_list:
                if dataset_name in res["video_path"]:
                    total[dataset_name].append(value)
                    total["avg"].append(value)
                    break
            
        for key in total:
            value = np.mean(np.array(total[key])).item()
            metrics[key][item] = value

with open(output_json, "w") as fp:
    json.dump(metrics, fp, indent=2)