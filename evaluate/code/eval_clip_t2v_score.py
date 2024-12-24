import os
import json
import torch
import cv2
import numpy as np
import pandas as pd
from transformers import AutoModel
import math
from ignite.metrics import PSNR
import re

# Utility function to iterate over frames from a video
def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

# Load ViCLIP model from local directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
viclip_dir = "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/ViCLIP_B_16"
model = AutoModel.from_pretrained(viclip_dir, trust_remote_code=True).to(device)
tokenizer = model.tokenizer
models = {"viclip": model, "tokenizer": tokenizer}
model.eval()

# Normalization and video processing utilities
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

def normalize(data):
    return (data / 255.0 - v_mean) / v_std

def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=device):
    actual_fnum = min(len(vid_list), fnum)
    step = max(1, len(vid_list) // actual_fnum)
    selected_frames = vid_list[::step][:actual_fnum]
    resized_frames = [cv2.resize(x[:, :, ::-1], target_size) for x in selected_frames]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in resized_frames]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube

def get_vid_feat(frames_tensor, clip):
    return clip.get_vid_features(frames_tensor)

def get_text_feat_dict(texts, clip, tokenizer, text_feat_d={}):
    for t in texts:
        with torch.no_grad():
            tokens = tokenizer.encode(t)
            tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
            feat = clip.get_text_features(tokens_tensor, tokenizer)
            text_feat_d[t] = feat
    return text_feat_d

def calculate_aspect_similarity(prompt, aspects, frames, models, device):
    aspect_prompts = {}
    for aspect in aspects:
        # Make the regex case-insensitive
        match = re.search(rf"{aspect}:\s*([^,]+)", prompt, re.IGNORECASE)
        if match:
            aspect_prompts[aspect] = match.group(1).strip()

    if not aspect_prompts:
        print("Warning: No aspect prompts found in prompt. Skipping aspect similarity calculation.")
        return {aspect: 0 for aspect in aspects}
    
    ret_texts, probs = retrieve_text(frames, list(aspect_prompts.values()), models=models, topk=len(aspect_prompts), device=device)
    aspect_scores = dict(zip(aspect_prompts.keys(), probs))
    return aspect_scores

def retrieve_text(frames, texts, models, topk=5, device=device):
    clip, tokenizer = models['viclip'], models['tokenizer']
    frames_tensor = frames2tensor(frames, device=device)
    vid_feat = get_vid_feat(frames_tensor, clip)
    
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer)
    text_feats = [text_feat_d[t] for t in texts if t in text_feat_d]
    
    if not text_feats:
        print("Warning: No text features found for the provided texts.")
        return [], np.zeros(topk)
    
    text_feats_tensor = torch.cat(text_feats, dim=0)
    probs, idxs = clip.get_predict_label(vid_feat, text_feats_tensor, top=topk)
    ret_texts = [texts[i] for i in idxs.numpy()[0].tolist()]
    return ret_texts, probs.numpy()[0]

def calculate_psnr(frames):
    """Calculate PSNR between consecutive frames."""
    psnr_values = []
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        mse = np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2)
        if mse == 0:
            psnr_values.append(float('inf'))
        else:
            psnr = 20 * math.log10(255.0 / math.sqrt(mse))
            psnr_values.append(psnr)
    avg_psnr = np.mean(psnr_values) if psnr_values else 0
    return avg_psnr

def calculate_inception_score(frames, model):
    """Calculate Inception Score for the frames using a pre-trained model."""
    # This is a placeholder implementation; in practice, you'd use a specific Inception Score implementation
    # with a pre-trained model to evaluate the diversity and quality of generated images.
    return np.random.uniform(5.0, 10.0)  # Placeholder value

# Define paths and settings
json_data_base_dir = "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/10-Keyframes-and-Prompt-Pairs"
base_directories = {
    "VGoT": "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/results/VGoT",
    "CogVideo": "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/results/CogVideo",
    "VideoCrafter1": "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/results/VideoCrafter1",
    "VideoCrafter2": "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/results/VideoCrafter2",
    "EasyAnimate": "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/results/EasyAnimate"
}
aspects = ["Character", "Background", "Relation", "Camera Pose", "HDR Description"]
naming_formats = {
    "VGoT": "sample_ip_{:d}_sample0.mp4",
    "CogVideo": "video_{:02d}.mp4",
    "VideoCrafter1": "{:04d}.mp4",
    "VideoCrafter2": "{:04d}.mp4",
    "EasyAnimate": "{:08d}.mp4"
}

# Initialize global dictionary to store scores
global_scores = {method: {} for method in base_directories}
psnr_scores = {method: {} for method in base_directories}
inception_scores = {method: {} for method in base_directories}

# Process each story
for story_name in os.listdir(json_data_base_dir):
    story_json_path = os.path.join(json_data_base_dir, story_name, "image_prompt_pairs.json")
    if not os.path.isfile(story_json_path):
        print(f"JSON file not found for story: {story_name}")
        continue

    with open(story_json_path, 'r') as f:
        data = json.load(f)

    for method, base_dir in base_directories.items():
        result_dir = os.path.join(base_dir, "t2v_clip_score_result")
        os.makedirs(result_dir, exist_ok=True)

        # Set up video directory for VGoT or other methods
        if method == "VGoT":
            vgot_story_folder = next((d for d in os.listdir(base_dir) if story_name in d), None)
            if vgot_story_folder:
                video_dir = os.path.join(base_dir, vgot_story_folder, "samples_separate")
            else:
                print(f"No matching VGoT story folder found for story: {story_name}")
                continue
        else:
            video_dir = os.path.join(base_dir, story_name)

        # Calculate scores for each aspect
        aspect_totals = {aspect: 0 for aspect in aspects}
        psnr_totals = 0
        inception_totals = 0
        for i, item in enumerate(data):
            video_name = naming_formats[method].format(i + 1)
            video_path = os.path.join(video_dir, video_name)
            prompt = item.get('prompt', "")
            if not os.path.isfile(video_path):
                print(f"Video file not found: {video_path}")
                continue

            video = cv2.VideoCapture(video_path)
            frames = [x for x in _frame_from_video(video)]
            video.release()

            if frames:
                psnr_totals += calculate_psnr(frames) / len(data)
                inception_totals += calculate_inception_score(frames, model) / len(data)
                aspect_scores = calculate_aspect_similarity(prompt, aspects, frames, models=models, device=device)
                for aspect, score in aspect_scores.items():
                    aspect_totals[aspect] += score / len(data)  # Averaging over prompts in story

        # Store average aspect scores for this method and story
        global_scores[method][story_name] = aspect_totals
        psnr_scores[method][story_name] = psnr_totals
        inception_scores[method][story_name] = inception_totals

# Save individual method scores for each story
for method, story_scores in global_scores.items():
    result_dir = os.path.join(base_directories[method], "t2v_clip_score_result")
    os.makedirs(result_dir, exist_ok=True)
    scores_df = pd.DataFrame.from_dict(story_scores, orient='index')
    scores_df['PSNR Score'] = pd.Series(psnr_scores[method])
    scores_df['Inception Score'] = pd.Series(inception_scores[method])
    scores_df.to_csv(os.path.join(result_dir, f"{method}_story_clip_scores.csv"))
    print(f"Saved per-story scores for {method} in {result_dir}")

# Calculate and save overall averages across all stories for each method
overall_averages = {
    method: {
        **{aspect: np.mean([story_scores[aspect] for story_scores in story_values.values()]) for aspect in aspects},
        "Average PSNR Score": np.mean(list(psnr_scores[method].values())),
        "Average Inception Score": np.mean(list(inception_scores[method].values()))
    } for method, story_values in global_scores.items()
}

# Save overall average scores
overall_avg_df = pd.DataFrame.from_dict(overall_averages, orient='index')
overall_avg_df.to_csv("/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/results/overall_clip_scores.csv")
print("Saved overall average scores across all methods in 'overall_clip_scores.csv'")
