export CUDA_VISIBLE_DEVICES=2

/data/yyang/miniconda3/envs/magiclidar/bin/python main.py \
--output_dir /dataset/yyang/magiclidar/log/moefusion_fusion_alan \
--modality fusion \
--attribute fusion \
--moe_fusion \
--batch_size 2
# --resume /dataset/yyang/magiclidar/log/moefusion_image_alan_rerun/checkpoint0013.pth