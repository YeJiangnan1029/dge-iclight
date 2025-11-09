import yaml
import json
from pathlib import Path
from qwen_vqa import QwenVQAModel
from PIL import Image

def construct_benchmark(
    config_path: str = "data_source.yaml",
    output_path: str = "scene_prompt.json"
):
    # 1. 加载配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        data_sources = yaml.safe_load(f)

    results = {}

    # 2. 加载模型
    with QwenVQAModel() as model:
        for dataset_name, dataset_info in data_sources.items():
            base_path = Path(dataset_info["base_path"])
            image_rel_path = dataset_info["image_pick"]

            for scene in dataset_info["scenes"]:
                image_path = base_path / scene / image_rel_path
                if not image_path.exists() and image_path.suffix.lower() == ".png":
                    alt_path = image_path.with_suffix(".jpg")
                    if alt_path.exists():
                        image_path = alt_path
                if not image_path.exists():
                    print(f"[WARN] Image not found: {image_path}")
                    continue

                print(f"[INFO] Processing {dataset_name}/{scene} ...")

                try:
                    image = Image.open(image_path).convert("RGB")
                    # 3. 使用 QwenVQAModel 生成 prompt
                    prompt_query = "Give me 3 text prompts to relight this image."
                    output = model.inference(image, prompt_query)

                    # 输出格式应是 ["p1", "p2", "p3"]
                    try:
                        prompts = json.loads(output.strip())
                        if not isinstance(prompts, list) or len(prompts) != 3:
                            raise ValueError
                    except Exception:
                        print(f"[WARN] Invalid output format for {dataset_name}/{scene}: {output}")
                        prompts = [output]

                    # 4. 保存结果
                    key = f"{dataset_name}/{scene}"
                    results[key] = prompts

                except Exception as e:
                    print(f"[ERROR] Failed processing {dataset_name}/{scene}: {e}")

    # 5. 写出 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Benchmark constructed and saved to {output_path}")


if __name__ == "__main__":
    construct_benchmark(
        config_path = "script/benchmark/data_source.yaml", 
        output_path = "script/benchmark/scene_prompt.json"
    )
