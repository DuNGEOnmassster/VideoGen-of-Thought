seed=123
story_name=Mary

ckpt=/root/mingzhe/pretrained_weights/pretrained/Doubiiu/DynamiCrafter_1024/model.ckpt
config=configs/inference_module3.yaml

prompt_dir=/root/VideoGen-of-Thought/data/${story_name}
keyframe_dir=/root/VideoGen-of-Thought/KeyFrames/${story_name}
res_dir=/root/VideoGen-of-Thought/Shot_Videos/${story_name}

CUDA_VISIBLE_DEVICES=0 python3 scripts/generate_module4-copy.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$story_name \
--n_samples 1 \
--bs 1 --height 576 --width 1024 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--keyframe_dir $keyframe_dir \
--text_input \
--video_length 16 \
--frame_stride 10 \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae


## multi-cond CFG: the <unconditional_guidance_scale> is s_txt, <cfg_img> is s_img
#--multiple_cond_cfg --cfg_img 7.5
#--loop