python launch.py --config configs/dge.yaml --train --gpu 0 \
    trainer.max_steps=1500 system.prompt_processor.prompt="Let the sunlight shine on the man's face" \
    data.source="/mnt/16T/yejiangnan/work/cvpr25_EditSplat/dataset/face" \
    system.guidance.guidance_scale=10.0 \
    system.gs_source="/mnt/16T/yejiangnan/work/cvpr25_EditSplat/output/3dgs/face/point_cloud/iteration_30000/point_cloud.ply"