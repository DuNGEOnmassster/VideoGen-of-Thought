# Generate new story
CUDA_VISIBLE_DEVICES=1 python scripts/cross_shot_propagation.py \
    --user_input "a young inventor creates a time machine in her garage" \
    --story_name "Time_Inventor" \
    --num_shot 18 \
    --story_type 3

# # Use existing prompt
# python scripts/cross_shot_propagation.py \
#     --use_exist_prompt "asset/story_type1/Mary" \
#     --keyframe_path "KeyFrames/Mary" \
#     --shot_save_path "Shot_Videos/Mary"