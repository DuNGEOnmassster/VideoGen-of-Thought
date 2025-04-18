#!/bin/bash

# --- VGoT Configuration ---
USER_INPUT="the journey of a boy named Alex who grows up in a small fishing village and becomes a renowned marine biologist."
STORY_NAME="Alex_Test"  # Changed name slightly to avoid overwriting previous results
STORY_TYPE=1
SEED=42 # Changed seed slightly
NUM_SHOTS=5 # Reduced for faster testing

# --- FramePack I2V Configuration ---
SHOT_DURATION_SECONDS=4.0
FP_STEPS=25
FP_GS=10.0
FP_N_PROMPT="low quality, worst quality, blurry, motion blur, text, watermark, signature, bad anatomy, bad hands"
FP_USE_TEACACHE=true # Set to false to disable: --fp_no_teacache
FP_RESOLUTION=640

# --- Output Paths ---
BASE_ASSET_PATH="asset/round5" # Root directory for all generated assets

# --- Execution ---
SCRIPT_PATH="scripts/videogen_of_thought_fp.py" # Ensure this is the correct path
CUDA_VISIBLE_DEVICES=6

# # Construct command
# CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $SCRIPT_PATH \
#     --user_input \"$USER_INPUT\" \
#     --story_name \"$STORY_NAME\" \
#     --story_type $STORY_TYPE \
#     --seed $SEED \
#     --num_shot $NUM_SHOTS \
#     --base_path \"$BASE_ASSET_PATH\" \
#     --fp_shot_duration_seconds $SHOT_DURATION_SECONDS \
#     --fp_steps $FP_STEPS \
#     --fp_gs $FP_GS \
#     --fp_n_prompt \"$FP_N_PROMPT\" \
#     --fp_resolution $FP_RESOLUTION \
#     --save_individual"

# Construct command
CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $SCRIPT_PATH \
    --use_exist_prompt asset/round5/seed42/story_type1/Alex_Test \
    --seed $SEED \
    --num_shot $NUM_SHOTS \
    --base_path \"$BASE_ASSET_PATH\" \
    --fp_shot_duration_seconds $SHOT_DURATION_SECONDS \
    --fp_steps $FP_STEPS \
    --fp_gs $FP_GS \
    --fp_n_prompt \"$FP_N_PROMPT\" \
    --fp_resolution $FP_RESOLUTION \
    --save_individual"

# Add TeaCache flag
if [ "$FP_USE_TEACACHE" = true ]; then
    CMD="$CMD --fp_use_teacache"
else
    CMD="$CMD --fp_no_teacache"
fi

# Print and Execute
echo "Running VGoT with FramePack I2V:"
echo "$CMD"
eval "$CMD"

echo "Script execution finished."