
### final 
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/final_inference.py --json_file Prompts/MorphBench/morphbench_animation_ours.json \
 --ckpt_path checkpoints/dynamicrafter_512_interp_v1/model.ckpt --config configs/inference_512_v1.0_tvg.yaml \
 --savedir abl_results/ours/MorphBench/animation --n_samples 1 --bs 1 --height 512 --width 512 --unconditional_guidance_scale 7.5 \
 --ddim_steps 10 --ddim_eta 1.0 --text_input --video_length 16 --frame_stride 16 --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
 --perframe_ae --interp

CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/final_inference.py --json_file Prompts/MorphBench/morphbench_metamorphosis_ours.json \
 --ckpt_path checkpoints/dynamicrafter_512_interp_v1/model.ckpt --config configs/inference_512_v1.0_tvg.yaml \
 --savedir abl_results/ours/MorphBench/metamorphosis --n_samples 1 --bs 1 --height 512 --width 512 --unconditional_guidance_scale 7.5 \
 --ddim_steps 10 --ddim_eta 1.0 --text_input --video_length 16 --frame_stride 16 --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
 --perframe_ae --interp

CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/final_inference.py --json_file Prompts/TC-Bench-I2V/ours.json \
 --ckpt_path checkpoints/dynamicrafter_512_interp_v1/model.ckpt --config configs/inference_512_v1.0_tvg.yaml \
 --savedir abl_results/ours/TC_Bench_I2V/ --n_samples 1 --bs 1 --height 320 --width 512 --unconditional_guidance_scale 7.5 \
 --ddim_steps 10 --ddim_eta 1.0 --text_input --video_length 16 --frame_stride 16 --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
 --perframe_ae --interp


CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/final_inference.py --json_file Prompts/qualitative_results/prompts_320.json \
 --ckpt_path checkpoints/dynamicrafter_512_interp_v1/model.ckpt --config configs/inference_512_v1.0_tvg.yaml \
 --savedir abl_results/ours/Qualitative/ --n_samples 1 --bs 1 --height 320 --width 512 --unconditional_guidance_scale 7.5 \
 --ddim_steps 10 --ddim_eta 1.0 --text_input --video_length 16 --frame_stride 16 --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
 --perframe_ae --interp

CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/final_inference.py --json_file Prompts/qualitative_results/prompts_512.json \
 --ckpt_path checkpoints/dynamicrafter_512_interp_v1/model.ckpt --config configs/inference_512_v1.0_tvg.yaml \
 --savedir abl_results/ours/Qualitative/ --n_samples 1 --bs 1 --height 512 --width 512 --unconditional_guidance_scale 7.5 \
 --ddim_steps 10 --ddim_eta 1.0 --text_input --video_length 16 --frame_stride 16 --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
 --perframe_ae --interp

### DynamiCrafter+ Slerp+Guass

 CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/singelway_inference.py --json_file Prompts/MorphBench/morphbench_animation_ours.json \
 --ckpt_path checkpoints/dynamicrafter_512_interp_v1/model.ckpt --config configs/inference_512_v1.0_tvg_single.yaml \
 --savedir abl_results/dynamicrafter_text_guass/MorphBench/animation --n_samples 1 --bs 1 --height 512 --width 512 --unconditional_guidance_scale 7.5 \
 --ddim_steps 10 --ddim_eta 1.0 --text_input --video_length 16 --frame_stride 16 --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
 --perframe_ae --interp


 CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/singelway_inference.py --json_file Prompts/MorphBench/morphbench_metamorphosis_ours.json \
 --ckpt_path checkpoints/dynamicrafter_512_interp_v1/model.ckpt --config configs/inference_512_v1.0_tvg_single.yaml \
 --savedir abl_results/dynamicrafter_text_guass/MorphBench/metamorphosis --n_samples 1 --bs 1 --height 512 --width 512 --unconditional_guidance_scale 7.5 \
 --ddim_steps 10 --ddim_eta 1.0 --text_input --video_length 16 --frame_stride 16 --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
 --perframe_ae --interp


### DynamiCrafter+ Slerp
 CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/singelway_inference.py --json_file Prompts/MorphBench/morphbench_animation_ours.json \
 --ckpt_path checkpoints/dynamicrafter_512_interp_v1/model.ckpt --config configs/inference_512_v1.0.yaml \
 --savedir abl_results/dynamicrafter_text/MorphBench/animation --n_samples 1 --bs 1 --height 512 --width 512 --unconditional_guidance_scale 7.5 \
 --ddim_steps 10 --ddim_eta 1.0 --text_input --video_length 16 --frame_stride 16 --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
 --perframe_ae --interp


 CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/singelway_inference.py --json_file Prompts/MorphBench/morphbench_metamorphosis_ours.json \
 --ckpt_path checkpoints/dynamicrafter_512_interp_v1/model.ckpt --config configs/inference_512_v1.0.yaml \
 --savedir abl_results/dynamicrafter_text/MorphBench/metamorphosis --n_samples 1 --bs 1 --height 512 --width 512 --unconditional_guidance_scale 7.5 \
 --ddim_steps 10 --ddim_eta 1.0 --text_input --video_length 16 --frame_stride 16 --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
 --perframe_ae --interp

### DynamiCrafter
 CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/base_inference.py --json_file Prompts/MorphBench/morphbench_animation_originalmodel.json \
 --ckpt_path checkpoints/dynamicrafter_512_interp_v1/model.ckpt --config configs/inference_512_v1.0.yaml \
 --savedir abl_results/dynamicrafter_512_interp_v1/MorphBench/animation --n_samples 1 --bs 1 --height 512 --width 512 --unconditional_guidance_scale 7.5 \
 --ddim_steps 10 --ddim_eta 1.0 --text_input --video_length 16 --frame_stride 16 --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
 --perframe_ae --interp

 CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/base_inference.py --json_file Prompts/MorphBench/morphbench_metamorphosis_originalmodel.json \
 --ckpt_path checkpoints/dynamicrafter_512_interp_v1/model.ckpt --config configs/inference_512_v1.0.yaml \
 --savedir abl_results/dynamicrafter_512_interp_v1/MorphBench/metamorphosis --n_samples 1 --bs 1 --height 512 --width 512 --unconditional_guidance_scale 7.5 \
 --ddim_steps 10 --ddim_eta 1.0 --text_input --video_length 16 --frame_stride 16 --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
 --perframe_ae --interp