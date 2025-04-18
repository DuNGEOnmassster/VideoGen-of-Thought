#!/bin/bash

# --- VGoT Configuration ---
USER_INPUT="the journey of a boy named Alex who grows up in a small fishing village and becomes a renowned marine biologist."
STORY_NAME="Alex_FP_Test"  # Changed name slightly to avoid overwriting previous results
STORY_TYPE=1
SEED=43 # Changed seed slightly
NUM_SHOTS=5 # Reduced for faster testing

# --- FramePack I2V Configuration ---
SHOT_DURATION_SECONDS=4.0
FP_STEPS=25
FP_GS=10.0
FP_N_PROMPT="low quality, worst quality, blurry, motion blur, text, watermark, signature, bad anatomy, bad hands"
FP_USE_TEACACHE=true # Set to false to disable: --fp_no_teacache
FP_RESOLUTION=640

# --- Model Paths (MAKE SURE THESE ARE CORRECT FOR YOUR SYSTEM) ---
FP_HF_MODEL_DIR="weights/HunyuanVideo"
FP_FRAMEPACK_MODEL_DIR="weights/FramePackI2V_HY"
FP_SIGLIP_MODEL_DIR="weights/flux_redux_bfl"
# Kolors/IP-Adapter paths are assumed to be hardcoded correctly inside the python script's functions (prepare_avatar_model, prepare_keyframe_model)

# --- Output Paths ---
BASE_ASSET_PATH="asset/round5" # Root directory for all generated assets


# --- Execution ---
SCRIPT_PATH="scripts/videogen_of_thought_fp.py" # Ensure this is the correct path
CUDA_VISIBLE_DEVICES=7

# Construct command
CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $SCRIPT_PATH \
    --user_input \"$USER_INPUT\" \
    --story_name \"$STORY_NAME\" \
    --story_type $STORY_TYPE \
    --seed $SEED \
    --num_shot $NUM_SHOTS \
    --base_path \"$BASE_ASSET_PATH\" \
    --fp_shot_duration_seconds $SHOT_DURATION_SECONDS \
    --fp_steps $FP_STEPS \
    --fp_gs $FP_GS \
    --fp_n_prompt \"$FP_N_PROMPT\" \
    --fp_resolution $FP_RESOLUTION \
    --fp_hf_model_dir \"$FP_HF_MODEL_DIR\" \
    --fp_framepack_model_dir \"$FP_FRAMEPACK_MODEL_DIR\" \
    --fp_siglip_model_dir \"$FP_SIGLIP_MODEL_DIR\""

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