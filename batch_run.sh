#!/bin/bash

# base dir
BASEDIR="/homeC/Public/yjn_data"
CONFIG="configs/dge.yaml"
EXP_ROOT="$BASEDIR/dge_iclight_output"
SOURCE_BASE="$BASEDIR/mipnerf360"
GS_BASE="$BASEDIR/3dgs_output/mipnerf360"

# 光照方向
DIRECTIONS=("left" "right" "top" "bottom" "front")

# 场景列表: name object prompt_core
SCENES=(
  "bicycle|bicycle|detailed structure, sunshine, outdoor, warm atmosphere, sunlight from"
  "bonsai|bonsai|detailed leaves, sunshine, indoor, natural atmosphere, sunlight from"
  "counter|table|detailed surface, warm atmosphere, indoor, sunlight from"
  "flowers|flowers|detailed petals, sunshine, outdoor, warm atmosphere, sunlight from"
  "garden|wooden table|detailed texture, sunset over garden, outdoor, warm atmosphere, sunlight from"
  "kitchen|lego|detailed structure, indoor, warm atmosphere, sunlight from"
  "room|television|detailed screen, shadow from window, indoor, sunlight from"
  "stump|stamp|detailed texture, natural lighting, outdoor, sunlight from"
  "treehill|tree|detailed leaves, sunshine, outdoor, warm atmosphere, sunlight from"
)

for scene in "${SCENES[@]}"; do
  IFS="|" read -r NAME OBJECT PROMPT_CORE <<< "$scene"

  for dir in "${DIRECTIONS[@]}"; do
    PROMPT="${PROMPT_CORE} ${dir}"
    echo "Launching training for $NAME with light from $dir ..."
    python launch.py --config "$CONFIG" \
      --train data.source="$SOURCE_BASE/$NAME" \
      system.gs_source="$GS_BASE/$NAME/point_cloud/iteration_30000/point_cloud.ply" \
      exp_root_dir="$EXP_ROOT" \
      system.cache_dir="$GS_BASE/$NAME/cache" \
      system.prompt_processor.prompt="$PROMPT" \
      name="dge-0913"
  done
done
