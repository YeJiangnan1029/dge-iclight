from datetime import datetime
import os
import csv
from time import time
import pandas as pd

def collect_metrics(exp_root, output_csv="summary.csv"):
    results = []

    for method in os.listdir(exp_root):
        method_dir = os.path.join(exp_root, method)
        if not os.path.isdir(method_dir):
            continue

        for exp_name in os.listdir(method_dir):
            exp_dir = os.path.join(method_dir, exp_name)
            csv_path = os.path.join(exp_dir, "csv_logs", "version_0", "metrics.csv")

            if not os.path.exists(csv_path):
                continue

            # 解析 text_prompt
            if "@" in exp_name:
                text = exp_name.split("@")[0]
            else:
                text = exp_name
            text_prompt = text.replace("_", " ")

            # 读取 CSV
            try:
                df = pd.read_csv(csv_path)
                if len(df) < 2:
                    continue

                last_row = df.iloc[-1]
                second_last_row = df.iloc[-2]

                test_clip_score = last_row.get("test/clip_score", None)
                test_clip_dir_score = last_row.get("test/clip_directional_score", None)
                val_lpips = second_last_row.get("val/lpips", None)
                val_psnr = second_last_row.get("val/psnr", None)
                val_ssim = second_last_row.get("val/ssim", None)

                results.append({
                    "method": method,
                    "text_prompt": text_prompt,
                    "test/clip_score": test_clip_score,
                    "test/clip_directional_score": test_clip_dir_score,
                    "val/lpips": val_lpips,
                    "val/psnr": val_psnr,
                    "val/ssim": val_ssim
                })
            except Exception as e:
                print(f"读取 {csv_path} 出错: {e}")

    # 保存结果
    if results:
        keys = results[0].keys()
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

        print(f"汇总完成，结果保存至 {output_csv}")
    else:
        print("未找到任何 metrics.csv 文件")

if __name__ == "__main__":
    exp_root = "/homeC/Public/yjn_data/dge_iclight_output"  # 修改为你的实验根目录
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"summary_{timestamp_str}.csv"
    collect_metrics(exp_root, output_csv)
