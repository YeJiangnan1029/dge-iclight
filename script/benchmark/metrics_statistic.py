import os
import json
from collections import defaultdict

def aggregate_metrics(base_folder):
    # 定义三个数据集类别及一个总类别
    categories = {
        "in2n": [],
        "mip360": [],
        "scannetpp": [],
        "all": []
    }

    # 遍历 scene_name / exp_name / save/metrics.json
    for scene_name in os.listdir(base_folder):
        scene_path = os.path.join(base_folder, scene_name)
        if not os.path.isdir(scene_path):
            continue
        
        for exp_name in os.listdir(scene_path):
            metrics_path = os.path.join(scene_path, exp_name, "save", "metrics.json")
            if not os.path.exists(metrics_path):
                continue

            try:
                with open(metrics_path, "r") as f:
                    data = json.load(f)
                
                # 判断属于哪个类别
                matched = False
                for key in ["in2n", "mip360", "scannetpp"]:
                    if key in scene_name.lower():
                        categories[key].append(data)
                        matched = True
                        break
                if not matched:
                    # 若不属于任何类别，可选择是否统计入all
                    pass

                # 所有文件都统计入总类
                categories["all"].append(data)
                print(f"Loaded: {scene_name}/{exp_name}")
            except Exception as e:
                print(f"⚠️ Error reading {metrics_path}: {e}")

    # 输出结果
    print("\n========== Average Metrics ==========")
    for key, metric_list in categories.items():
        if len(metric_list) == 0:
            print(f"\n[{key}] No data found.")
            continue

        print(f"\n[{key}] ({len(metric_list)} experiments)")
        sums = defaultdict(float)
        for m in metric_list:
            for k, v in m.items():
                sums[k] += v

        for k in sums.keys():
            avg = sums[k] / len(metric_list)
            print(f"{k:25s}: {avg:.6f}")

if __name__ == "__main__":
    base_folder = "output/dge_benchmark/dge_ip2p"
    aggregate_metrics(base_folder)
