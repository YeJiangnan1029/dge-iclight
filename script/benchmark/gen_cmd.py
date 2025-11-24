import yaml
import json
from pathlib import Path

# ----------------------------
# 文件路径
# ----------------------------
data_source_path = "script/benchmark/data_source.yaml"
scene_prompt_path = "script/benchmark/scene_prompt_dir.json"
output_sh_path = "pickup_mv.sh"

# ----------------------------
# 读取配置
# ----------------------------
with open(data_source_path, "r") as f:
    data_source = yaml.safe_load(f)

with open(scene_prompt_path, "r") as f:
    scene_prompts = json.load(f)

# ----------------------------
# 写入 bash 脚本头部
# ----------------------------
lines = [
    "#!/bin/bash\n",
    "export CUDA_VISIBLE_DEVICES=0\n",
    "\n"
]

exp_root_dir = "/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_ablation_mv/mv"

# ----------------------------
# 遍历数据源与场景
# ----------------------------
for source_name, source_cfg in data_source.items():
    if source_name != "in2n": continue  # 仅处理 in2n 数据源
    base_path = source_cfg.get("base_path", "")
    gs_base_path = source_cfg.get("gs_base_path", base_path)
    image_pick = source_cfg.get("image_pick", "")
    gs_ply = source_cfg.get("gs_ply", "")
    cache_path = source_cfg.get("cache_path", "")
    scenes = source_cfg.get("scenes", [])
    exp_folder = source_cfg.get("exp_folder", None)
    scene_sub_folder = source_cfg.get("scene_subfolder", "")

    for scene in scenes:
        full_scene_name = f"{source_name}/{scene}"
        prompts = scene_prompts.get(full_scene_name, [])
        if not prompts:
            print(f"[Warning] Missing prompts for scene: {full_scene_name}")
            continue

        for idx, prompt in enumerate(prompts):
            safe_prompt = (
                prompt.replace(" ", "_")
                .replace(",", "")
                .replace('"', "")
                .replace("'", "")
            )
            exp_name = f"dge-iclight_{source_name}_{scene}_{idx}"

            data_source_dir = f"{base_path}/{scene}"
            if scene_sub_folder:
                data_source_dir = f"{data_source_dir}/{scene_sub_folder}"
            gs_source_path = f"{gs_base_path}/{scene}/{gs_ply}"
            cache_dir = f"{gs_base_path}/{scene}/{cache_path}"
            if exp_folder:
                exp_dir = f"{gs_base_path}/{scene}/{exp_folder}"
            else:
                exp_dir = ""

            cmd = f"""\
echo "--------------------------------------------"
echo "Launching: {exp_name}"
echo "Scene path: {data_source_dir}"
echo "Prompt: {prompt}"
echo "--------------------------------------------"

python launch.py \\
    --config configs/dge.yaml \\
    --train \\
    data.source="{data_source_dir}" \\
    data.exp_folder="{exp_dir}" \\
    system.gs_source="{gs_source_path}" \\
    exp_root_dir="{exp_root_dir}" \\
    system.cache_dir="{cache_dir}" \\
    system.prompt_processor.prompt="{prompt}" \\
    system.guidance_type="dge-iclight" \\
    name="{exp_name}" \\
    system.ref_id="0" \\
    data.max_view_num="30"

"""
            lines.append(cmd)

# ----------------------------
# 写出到 batch_run.sh
# ----------------------------
Path(output_sh_path).write_text("".join(lines))
Path(output_sh_path).chmod(0o755)

print(f"[OK] 已生成可执行脚本: {output_sh_path}")
