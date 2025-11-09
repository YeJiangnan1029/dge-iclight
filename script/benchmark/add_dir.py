import json
import random
from pathlib import Path

# -----------------------------
# 配置
# -----------------------------
input_json = Path("script/benchmark/scene_prompt.json")           # 输入文件路径
output_json = Path("script/benchmark/scene_prompt_dir.json")  # 输出文件路径
directions = ["right", "left", "top", "bottom"]  # 随机光照方向选项
probabilities = [0.35, 0.35, 0.15, 0.15]

# 可选的光照描述模板
direction_templates = {
    "right": "light from the right",
    "left": "light from the left",
    "top": "light from top",
    "bottom": "light from bottom"
}


# ----------------------------
# 读取原始prompt数据
# ----------------------------
with open(input_json, "r") as f:
    data = json.load(f)

new_data = {}

# ----------------------------
# 为每个prompt随机添加光照方向
# ----------------------------
for scene, prompt_list in data.items():
    new_prompt_list = []
    for prompt in prompt_list:
        direction = random.choices(directions, weights=probabilities, k=1)[0]
        direction_phrase = direction_templates[direction]
        # 加上光照描述（前后都可以，这里放在末尾）
        new_prompt = f"{prompt}, {direction_phrase}"
        new_prompt_list.append(new_prompt)
    new_data[scene] = new_prompt_list

# ----------------------------
# 保存结果
# ----------------------------
Path(output_json).parent.mkdir(parents=True, exist_ok=True)
with open(output_json, "w") as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)

print(f"✅ 新的带方向 prompt 已保存到 {output_json}")