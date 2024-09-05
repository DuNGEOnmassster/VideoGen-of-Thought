# Note to set the `--target_size` argument consistent with the checkpoint

cd lumina_mgpt

CUDA_VISIBLE_DEVICES=0 python -u demos/demo_image_generation.py \
--pretrained_path /remote_shome/zhengmz/pretrained_models/pretrained/Alpha-VLLM/Lumina-mGPT-7B-768 \
--target_size 768