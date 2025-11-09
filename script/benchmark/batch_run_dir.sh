#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_bear_0"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/bear"
echo "Prompt: bear, forest, sunlight filtering through trees, natural lighting, warm atmosphere, light from top"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/bear" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/bear/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/bear/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/bear/splatfacto/exp/cache" \
    system.prompt_processor.prompt="bear, forest, sunlight filtering through trees, natural lighting, warm atmosphere, light from top" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_bear_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_bear_1"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/bear"
echo "Prompt: bear, woodland, twilight glow, golden hour, soft light, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/bear" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/bear/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/bear/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/bear/splatfacto/exp/cache" \
    system.prompt_processor.prompt="bear, woodland, twilight glow, golden hour, soft light, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_bear_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_bear_2"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/bear"
echo "Prompt: bear, forest, moonlight through branches, cool tone, mysterious shadows, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/bear" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/bear/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/bear/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/bear/splatfacto/exp/cache" \
    system.prompt_processor.prompt="bear, forest, moonlight through branches, cool tone, mysterious shadows, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_bear_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_campsite-small_0"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/campsite-small"
echo "Prompt: camping tents, outdoor, sunset, golden hour, warm firelight, light from bottom"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/campsite-small" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/campsite-small/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/campsite-small/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/campsite-small/splatfacto/exp/cache" \
    system.prompt_processor.prompt="camping tents, outdoor, sunset, golden hour, warm firelight, light from bottom" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_campsite-small_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_campsite-small_1"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/campsite-small"
echo "Prompt: camping site, outdoor, night sky, moonlight, cool blue tone, light from top"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/campsite-small" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/campsite-small/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/campsite-small/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/campsite-small/splatfacto/exp/cache" \
    system.prompt_processor.prompt="camping site, outdoor, night sky, moonlight, cool blue tone, light from top" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_campsite-small_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_campsite-small_2"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/campsite-small"
echo "Prompt: camping tents, outdoor, early morning fog, soft natural light, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/campsite-small" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/campsite-small/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/campsite-small/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/campsite-small/splatfacto/exp/cache" \
    system.prompt_processor.prompt="camping tents, outdoor, early morning fog, soft natural light, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_campsite-small_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_face_0"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/face"
echo "Prompt: portrait, detailed face, sunshine from window, warm atmosphere, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/face" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/face/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/face/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/face/splatfacto/exp/cache" \
    system.prompt_processor.prompt="portrait, detailed face, sunshine from window, warm atmosphere, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_face_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_face_1"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/face"
echo "Prompt: portrait, detailed face, neon light, city night, cinematic tone, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/face" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/face/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/face/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/face/splatfacto/exp/cache" \
    system.prompt_processor.prompt="portrait, detailed face, neon light, city night, cinematic tone, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_face_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_face_2"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/face"
echo "Prompt: portrait, detailed face, candle light, low-key lighting, moody shadows, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/face" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/face/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/face/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/face/splatfacto/exp/cache" \
    system.prompt_processor.prompt="portrait, detailed face, candle light, low-key lighting, moody shadows, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_face_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_fangzhou-small_0"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/fangzhou-small"
echo "Prompt: young man, detailed face, natural lighting, outdoor, warm, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/fangzhou-small" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/fangzhou-small/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/fangzhou-small/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/fangzhou-small/splatfacto/exp/cache" \
    system.prompt_processor.prompt="young man, detailed face, natural lighting, outdoor, warm, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_fangzhou-small_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_fangzhou-small_1"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/fangzhou-small"
echo "Prompt: young man, detailed face, neon light, city reflection, cool tone, light from top"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/fangzhou-small" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/fangzhou-small/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/fangzhou-small/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/fangzhou-small/splatfacto/exp/cache" \
    system.prompt_processor.prompt="young man, detailed face, neon light, city reflection, cool tone, light from top" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_fangzhou-small_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_fangzhou-small_2"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/fangzhou-small"
echo "Prompt: young man, detailed face, sunlight through blinds, soft indoor atmosphere, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/fangzhou-small" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/fangzhou-small/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/fangzhou-small/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/fangzhou-small/splatfacto/exp/cache" \
    system.prompt_processor.prompt="young man, detailed face, sunlight through blinds, soft indoor atmosphere, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_fangzhou-small_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_farm-small_0"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/farm-small"
echo "Prompt: open field, outdoor, bright sunlight, summer noon, high contrast, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/farm-small" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/farm-small/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/farm-small/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/farm-small/splatfacto/exp/cache" \
    system.prompt_processor.prompt="open field, outdoor, bright sunlight, summer noon, high contrast, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_farm-small_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_farm-small_1"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/farm-small"
echo "Prompt: farmland, outdoor, cloudy weather, diffused lighting, calm mood, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/farm-small" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/farm-small/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/farm-small/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/farm-small/splatfacto/exp/cache" \
    system.prompt_processor.prompt="farmland, outdoor, cloudy weather, diffused lighting, calm mood, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_farm-small_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_farm-small_2"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/farm-small"
echo "Prompt: field, outdoor, starry night, silver moonlight, tranquil tone, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/farm-small" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/farm-small/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/farm-small/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/farm-small/splatfacto/exp/cache" \
    system.prompt_processor.prompt="field, outdoor, starry night, silver moonlight, tranquil tone, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_farm-small_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_person-small_0"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/person-small"
echo "Prompt: man, indoor, sunshine from window, warm tone, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/person-small" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/person-small/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/person-small/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/person-small/splatfacto/exp/cache" \
    system.prompt_processor.prompt="man, indoor, sunshine from window, warm tone, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_person-small_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_person-small_1"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/person-small"
echo "Prompt: man, indoor, neon reflection, city light, cinematic, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/person-small" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/person-small/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/person-small/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/person-small/splatfacto/exp/cache" \
    system.prompt_processor.prompt="man, indoor, neon reflection, city light, cinematic, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_person-small_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_in2n_person-small_2"
echo "Scene path: /mnt/16T/yejiangnan/data/in2n/person-small"
echo "Prompt: man, indoor, shadow from blinds, dramatic lighting, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/in2n/person-small" \
    data.exp_folder="/mnt/16T/yejiangnan/data/3dgs_output/in2n/person-small/splatfacto/exp" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/in2n/person-small/splatfacto/exp/nerfstudio_models/splat.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/in2n/person-small/splatfacto/exp/cache" \
    system.prompt_processor.prompt="man, indoor, shadow from blinds, dramatic lighting, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_in2n_person-small_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_bicycle_0"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/bicycle"
echo "Prompt: bicycle, outdoor, morning sunlight, warm hue, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/bicycle" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/bicycle/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/bicycle/cache" \
    system.prompt_processor.prompt="bicycle, outdoor, morning sunlight, warm hue, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_bicycle_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_bicycle_1"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/bicycle"
echo "Prompt: bicycle, street night, neon reflections, cool blue tone, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/bicycle" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/bicycle/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/bicycle/cache" \
    system.prompt_processor.prompt="bicycle, street night, neon reflections, cool blue tone, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_bicycle_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_bicycle_2"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/bicycle"
echo "Prompt: bicycle, sunset, orange glow, long shadows, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/bicycle" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/bicycle/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/bicycle/cache" \
    system.prompt_processor.prompt="bicycle, sunset, orange glow, long shadows, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_bicycle_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_bonsai_0"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/bonsai"
echo "Prompt: bonsai tree, indoor, soft daylight from window, natural lighting, light from top"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/bonsai" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/bonsai/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/bonsai/cache" \
    system.prompt_processor.prompt="bonsai tree, indoor, soft daylight from window, natural lighting, light from top" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_bonsai_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_bonsai_1"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/bonsai"
echo "Prompt: bonsai tree, indoor, warm desk lamp illumination, cozy atmosphere, light from bottom"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/bonsai" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/bonsai/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/bonsai/cache" \
    system.prompt_processor.prompt="bonsai tree, indoor, warm desk lamp illumination, cozy atmosphere, light from bottom" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_bonsai_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_bonsai_2"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/bonsai"
echo "Prompt: bonsai tree, indoor, moonlight through window, tranquil cool tone, light from bottom"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/bonsai" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/bonsai/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/bonsai/cache" \
    system.prompt_processor.prompt="bonsai tree, indoor, moonlight through window, tranquil cool tone, light from bottom" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_bonsai_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_counter_0"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/counter"
echo "Prompt: kitchen counter, indoor, warm sunlight from window, morning atmosphere, light from bottom"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/counter" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/counter/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/counter/cache" \
    system.prompt_processor.prompt="kitchen counter, indoor, warm sunlight from window, morning atmosphere, light from bottom" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_counter_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_counter_1"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/counter"
echo "Prompt: kitchen counter, indoor, fluorescent light, evening cool tone, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/counter" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/counter/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/counter/cache" \
    system.prompt_processor.prompt="kitchen counter, indoor, fluorescent light, evening cool tone, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_counter_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_counter_2"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/counter"
echo "Prompt: kitchen counter, indoor, candle light, intimate mood, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/counter" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/counter/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/counter/cache" \
    system.prompt_processor.prompt="kitchen counter, indoor, candle light, intimate mood, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_counter_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_flowers_0"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/flowers"
echo "Prompt: flowers, outdoor, sunlight, bright and vivid colors, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/flowers" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/flowers/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/flowers/cache" \
    system.prompt_processor.prompt="flowers, outdoor, sunlight, bright and vivid colors, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_flowers_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_flowers_1"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/flowers"
echo "Prompt: flowers, outdoor, golden sunset light, soft focus, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/flowers" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/flowers/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/flowers/cache" \
    system.prompt_processor.prompt="flowers, outdoor, golden sunset light, soft focus, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_flowers_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_flowers_2"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/flowers"
echo "Prompt: flowers, outdoor, early morning dew, misty lighting, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/flowers" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/flowers/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/flowers/cache" \
    system.prompt_processor.prompt="flowers, outdoor, early morning dew, misty lighting, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_flowers_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_garden_0"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/garden"
echo "Prompt: garden, outdoor, sunlight filtering through leaves, warm tone, light from top"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/garden" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/garden/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/garden/cache" \
    system.prompt_processor.prompt="garden, outdoor, sunlight filtering through leaves, warm tone, light from top" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_garden_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_garden_1"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/garden"
echo "Prompt: garden, outdoor, rainy day, overcast lighting, muted colors, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/garden" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/garden/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/garden/cache" \
    system.prompt_processor.prompt="garden, outdoor, rainy day, overcast lighting, muted colors, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_garden_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_garden_2"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/garden"
echo "Prompt: garden, outdoor, night scene, moonlight and fireflies, magical atmosphere, light from bottom"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/garden" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/garden/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/garden/cache" \
    system.prompt_processor.prompt="garden, outdoor, night scene, moonlight and fireflies, magical atmosphere, light from bottom" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_garden_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_kitchen_0"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/kitchen"
echo "Prompt: kitchen, indoor, natural lighting from window, bright morning, light from top"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/kitchen" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/kitchen/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/kitchen/cache" \
    system.prompt_processor.prompt="kitchen, indoor, natural lighting from window, bright morning, light from top" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_kitchen_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_kitchen_1"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/kitchen"
echo "Prompt: kitchen, indoor, warm yellow lamp, cozy night atmosphere, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/kitchen" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/kitchen/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/kitchen/cache" \
    system.prompt_processor.prompt="kitchen, indoor, warm yellow lamp, cozy night atmosphere, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_kitchen_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_kitchen_2"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/kitchen"
echo "Prompt: kitchen, indoor, cool blue fluorescent light, modern tone, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/kitchen" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/kitchen/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/kitchen/cache" \
    system.prompt_processor.prompt="kitchen, indoor, cool blue fluorescent light, modern tone, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_kitchen_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_room_0"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/room"
echo "Prompt: living room, indoor, natural sunlight, soft shadows, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/room" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/room/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/room/cache" \
    system.prompt_processor.prompt="living room, indoor, natural sunlight, soft shadows, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_room_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_room_1"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/room"
echo "Prompt: living room, indoor, warm lamp light, cozy and calm, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/room" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/room/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/room/cache" \
    system.prompt_processor.prompt="living room, indoor, warm lamp light, cozy and calm, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_room_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_room_2"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/room"
echo "Prompt: living room, indoor, moonlight from window, blue tone, light from bottom"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/room" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/room/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/room/cache" \
    system.prompt_processor.prompt="living room, indoor, moonlight from window, blue tone, light from bottom" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_room_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_stump_0"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/stump"
echo "Prompt: tree stump, forest, sunlight through trees, golden time, light from bottom"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/stump" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/stump/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/stump/cache" \
    system.prompt_processor.prompt="tree stump, forest, sunlight through trees, golden time, light from bottom" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_stump_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_stump_1"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/stump"
echo "Prompt: tree stump, forest, rainy day, misty and soft lighting, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/stump" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/stump/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/stump/cache" \
    system.prompt_processor.prompt="tree stump, forest, rainy day, misty and soft lighting, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_stump_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_stump_2"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/stump"
echo "Prompt: tree stump, forest, moonlight, silvery cool tone, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/stump" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/stump/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/stump/cache" \
    system.prompt_processor.prompt="tree stump, forest, moonlight, silvery cool tone, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_stump_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_treehill_0"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/treehill"
echo "Prompt: tree on hill, outdoor, sunrise, orange glow, soft mist, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/treehill" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/treehill/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/treehill/cache" \
    system.prompt_processor.prompt="tree on hill, outdoor, sunrise, orange glow, soft mist, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_treehill_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_treehill_1"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/treehill"
echo "Prompt: tree on hill, outdoor, midday sun, bright lighting, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/treehill" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/treehill/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/treehill/cache" \
    system.prompt_processor.prompt="tree on hill, outdoor, midday sun, bright lighting, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_treehill_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_mip360_treehill_2"
echo "Scene path: /mnt/16T/yejiangnan/data/mipnerf360/treehill"
echo "Prompt: tree on hill, outdoor, sunset, purple sky, dramatic atmosphere, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/mipnerf360/treehill" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/treehill/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/mipnerf360/treehill/cache" \
    system.prompt_processor.prompt="tree on hill, outdoor, sunset, purple sky, dramatic atmosphere, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_mip360_treehill_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_0c5385e84b_0"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/0c5385e84b/dslr"
echo "Prompt: ceiling, indoor, sunlight reflection, warm white tone, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/0c5385e84b/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/0c5385e84b/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/0c5385e84b/cache" \
    system.prompt_processor.prompt="ceiling, indoor, sunlight reflection, warm white tone, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_0c5385e84b_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_0c5385e84b_1"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/0c5385e84b/dslr"
echo "Prompt: ceiling, indoor, artificial light, dim and moody, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/0c5385e84b/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/0c5385e84b/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/0c5385e84b/cache" \
    system.prompt_processor.prompt="ceiling, indoor, artificial light, dim and moody, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_0c5385e84b_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_0c5385e84b_2"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/0c5385e84b/dslr"
echo "Prompt: ceiling, indoor, morning daylight, natural soft tone, light from bottom"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/0c5385e84b/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/0c5385e84b/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/0c5385e84b/cache" \
    system.prompt_processor.prompt="ceiling, indoor, morning daylight, natural soft tone, light from bottom" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_0c5385e84b_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_5a269ba6fe_0"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/5a269ba6fe/dslr"
echo "Prompt: office, indoor, natural lighting, sunlight from window, light from top"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/5a269ba6fe/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/5a269ba6fe/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/5a269ba6fe/cache" \
    system.prompt_processor.prompt="office, indoor, natural lighting, sunlight from window, light from top" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_5a269ba6fe_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_5a269ba6fe_1"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/5a269ba6fe/dslr"
echo "Prompt: office, indoor, desk lamp light, warm tone, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/5a269ba6fe/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/5a269ba6fe/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/5a269ba6fe/cache" \
    system.prompt_processor.prompt="office, indoor, desk lamp light, warm tone, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_5a269ba6fe_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_5a269ba6fe_2"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/5a269ba6fe/dslr"
echo "Prompt: office, indoor, fluorescent light, cold corporate mood, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/5a269ba6fe/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/5a269ba6fe/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/5a269ba6fe/cache" \
    system.prompt_processor.prompt="office, indoor, fluorescent light, cold corporate mood, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_5a269ba6fe_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_08bbbdcc3d_0"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/08bbbdcc3d/dslr"
echo "Prompt: classroom, indoor, sunlight from window, bright and cheerful, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/08bbbdcc3d/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/08bbbdcc3d/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/08bbbdcc3d/cache" \
    system.prompt_processor.prompt="classroom, indoor, sunlight from window, bright and cheerful, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_08bbbdcc3d_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_08bbbdcc3d_1"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/08bbbdcc3d/dslr"
echo "Prompt: classroom, indoor, artificial light, soft evening tone, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/08bbbdcc3d/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/08bbbdcc3d/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/08bbbdcc3d/cache" \
    system.prompt_processor.prompt="classroom, indoor, artificial light, soft evening tone, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_08bbbdcc3d_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_08bbbdcc3d_2"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/08bbbdcc3d/dslr"
echo "Prompt: classroom, indoor, moonlight, blue tone, calm atmosphere, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/08bbbdcc3d/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/08bbbdcc3d/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/08bbbdcc3d/cache" \
    system.prompt_processor.prompt="classroom, indoor, moonlight, blue tone, calm atmosphere, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_08bbbdcc3d_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_39f36da05b_0"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/39f36da05b/dslr"
echo "Prompt: office, indoor, warm sunlight, golden tone, light from top"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/39f36da05b/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/39f36da05b/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/39f36da05b/cache" \
    system.prompt_processor.prompt="office, indoor, warm sunlight, golden tone, light from top" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_39f36da05b_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_39f36da05b_1"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/39f36da05b/dslr"
echo "Prompt: office, indoor, dim light, night atmosphere, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/39f36da05b/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/39f36da05b/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/39f36da05b/cache" \
    system.prompt_processor.prompt="office, indoor, dim light, night atmosphere, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_39f36da05b_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_39f36da05b_2"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/39f36da05b/dslr"
echo "Prompt: office, indoor, green tone, gothic Yharnam mood, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/39f36da05b/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/39f36da05b/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/39f36da05b/cache" \
    system.prompt_processor.prompt="office, indoor, green tone, gothic Yharnam mood, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_39f36da05b_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_56a0ec536c_0"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/56a0ec536c/dslr"
echo "Prompt: office, indoor, sunlight from window, bright workspace, light from bottom"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/56a0ec536c/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/56a0ec536c/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/56a0ec536c/cache" \
    system.prompt_processor.prompt="office, indoor, sunlight from window, bright workspace, light from bottom" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_56a0ec536c_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_56a0ec536c_1"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/56a0ec536c/dslr"
echo "Prompt: office, indoor, desk lamp light, shadowed corners, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/56a0ec536c/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/56a0ec536c/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/56a0ec536c/cache" \
    system.prompt_processor.prompt="office, indoor, desk lamp light, shadowed corners, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_56a0ec536c_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_56a0ec536c_2"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/56a0ec536c/dslr"
echo "Prompt: office, indoor, mixed lighting, sunlight and lamp blend, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/56a0ec536c/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/56a0ec536c/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/56a0ec536c/cache" \
    system.prompt_processor.prompt="office, indoor, mixed lighting, sunlight and lamp blend, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_56a0ec536c_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_a1d9da703c_0"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/a1d9da703c/dslr"
echo "Prompt: office, indoor, sunlight from ceiling window, natural tone, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/a1d9da703c/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/a1d9da703c/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/a1d9da703c/cache" \
    system.prompt_processor.prompt="office, indoor, sunlight from ceiling window, natural tone, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_a1d9da703c_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_a1d9da703c_1"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/a1d9da703c/dslr"
echo "Prompt: office, indoor, warm lamp lighting, cozy workspace, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/a1d9da703c/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/a1d9da703c/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/a1d9da703c/cache" \
    system.prompt_processor.prompt="office, indoor, warm lamp lighting, cozy workspace, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_a1d9da703c_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_a1d9da703c_2"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/a1d9da703c/dslr"
echo "Prompt: office, indoor, moonlight reflection, cool tone, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/a1d9da703c/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/a1d9da703c/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/a1d9da703c/cache" \
    system.prompt_processor.prompt="office, indoor, moonlight reflection, cool tone, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_a1d9da703c_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_bc2fce1d81_0"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/bc2fce1d81/dslr"
echo "Prompt: office, indoor, daylight from window, clear tone, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/bc2fce1d81/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/bc2fce1d81/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/bc2fce1d81/cache" \
    system.prompt_processor.prompt="office, indoor, daylight from window, clear tone, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_bc2fce1d81_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_bc2fce1d81_1"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/bc2fce1d81/dslr"
echo "Prompt: office, indoor, evening lamp light, golden hue, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/bc2fce1d81/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/bc2fce1d81/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/bc2fce1d81/cache" \
    system.prompt_processor.prompt="office, indoor, evening lamp light, golden hue, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_bc2fce1d81_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_bc2fce1d81_2"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/bc2fce1d81/dslr"
echo "Prompt: office, indoor, night fluorescent light, blue tone, light from the left"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/bc2fce1d81/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/bc2fce1d81/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/bc2fce1d81/cache" \
    system.prompt_processor.prompt="office, indoor, night fluorescent light, blue tone, light from the left" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_bc2fce1d81_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_dc263dfbf0_0"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/dc263dfbf0/dslr"
echo "Prompt: classroom, indoor, morning light, bright and clear, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/dc263dfbf0/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/dc263dfbf0/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/dc263dfbf0/cache" \
    system.prompt_processor.prompt="classroom, indoor, morning light, bright and clear, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_dc263dfbf0_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_dc263dfbf0_1"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/dc263dfbf0/dslr"
echo "Prompt: classroom, indoor, afternoon sunlight, golden hue, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/dc263dfbf0/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/dc263dfbf0/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/dc263dfbf0/cache" \
    system.prompt_processor.prompt="classroom, indoor, afternoon sunlight, golden hue, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_dc263dfbf0_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_dc263dfbf0_2"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/dc263dfbf0/dslr"
echo "Prompt: classroom, indoor, night light, dim and calm, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/dc263dfbf0/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/dc263dfbf0/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/dc263dfbf0/cache" \
    system.prompt_processor.prompt="classroom, indoor, night light, dim and calm, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_dc263dfbf0_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_ef18cf0708_0"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/ef18cf0708/dslr"
echo "Prompt: office, indoor, sunlight from window, bright tone, light from bottom"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/ef18cf0708/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/ef18cf0708/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/ef18cf0708/cache" \
    system.prompt_processor.prompt="office, indoor, sunlight from window, bright tone, light from bottom" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_ef18cf0708_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_ef18cf0708_1"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/ef18cf0708/dslr"
echo "Prompt: office, indoor, ceiling light, artificial illumination, dim, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/ef18cf0708/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/ef18cf0708/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/ef18cf0708/cache" \
    system.prompt_processor.prompt="office, indoor, ceiling light, artificial illumination, dim, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_ef18cf0708_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_ef18cf0708_2"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/ef18cf0708/dslr"
echo "Prompt: office, indoor, balanced mix of sunlight and ceiling light, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/ef18cf0708/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/ef18cf0708/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/ef18cf0708/cache" \
    system.prompt_processor.prompt="office, indoor, balanced mix of sunlight and ceiling light, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_ef18cf0708_2" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_fb564c935d_0"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/fb564c935d/dslr"
echo "Prompt: office desk, indoor, sunlight through blinds, striped shadows, light from bottom"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/fb564c935d/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/fb564c935d/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/fb564c935d/cache" \
    system.prompt_processor.prompt="office desk, indoor, sunlight through blinds, striped shadows, light from bottom" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_fb564c935d_0" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_fb564c935d_1"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/fb564c935d/dslr"
echo "Prompt: office desk, indoor, warm desk lamp light, cozy atmosphere, light from top"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/fb564c935d/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/fb564c935d/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/fb564c935d/cache" \
    system.prompt_processor.prompt="office desk, indoor, warm desk lamp light, cozy atmosphere, light from top" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_fb564c935d_1" \
    system.ref_id="0" \
    data.max_view_num="30"

echo "--------------------------------------------"
echo "Launching: dge-iclight_scannetpp_fb564c935d_2"
echo "Scene path: /mnt/16T/yejiangnan/data/scannetpp/fb564c935d/dslr"
echo "Prompt: office desk, indoor, night scene, cool fluorescent light, light from the right"
echo "--------------------------------------------"

python launch.py \
    --config configs/dge.yaml \
    --train \
    data.source="/mnt/16T/yejiangnan/data/scannetpp/fb564c935d/dslr" \
    data.exp_folder="" \
    system.gs_source="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/fb564c935d/point_cloud/iteration_30000/point_cloud.ply" \
    exp_root_dir="/mnt/16T/yejiangnan/work/dge-iclight/output/dge_benchmark/dge_iclight_dir" \
    system.cache_dir="/mnt/16T/yejiangnan/data/3dgs_output/scannetpp_3dgs/fb564c935d/cache" \
    system.prompt_processor.prompt="office desk, indoor, night scene, cool fluorescent light, light from the right" \
    system.guidance_type="dge-iclight" \
    name="dge-iclight_scannetpp_fb564c935d_2" \
    system.ref_id="0" \
    data.max_view_num="30"

