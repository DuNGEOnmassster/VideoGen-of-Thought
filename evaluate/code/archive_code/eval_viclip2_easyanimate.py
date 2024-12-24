import os
import json
import torch
import cv2
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
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

# Local ViCLIP directory path (update this to your actual path)
viclip_dir = "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/ViCLIP_B_16"

# Load the model and tokenizer from the local path
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
    if actual_fnum == 0:
        raise ValueError("No frames extracted from video.")
    
    step = max(1, len(vid_list) // actual_fnum)
    selected_frames = vid_list[::step][:actual_fnum]

    resized_frames = [cv2.resize(x[:, :, ::-1], target_size) for x in selected_frames]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in resized_frames]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()

    return vid_tube

# Extract video-level features using ViCLIP
def get_vid_feat(frames_tensor, clip):
    return clip.get_vid_features(frames_tensor)

# Extract text features using ViCLIP and tokenizer
def get_text_feat_dict(texts, clip, tokenizer, text_feat_d={}):
    for t in texts:
        with torch.no_grad():
            tokens = tokenizer.encode(t)
            tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
            feat = clip.get_text_features(tokens_tensor, tokenizer)
            text_feat_d[t] = feat
    return text_feat_d

# Calculate prompt aspect similarities
def calculate_aspect_similarity(prompt, aspects, frames, models, device):
    aspect_prompts = {}
    for aspect in aspects:
        match = re.search(rf"{aspect}:\s*([^,]+)", prompt)
        if match:
            aspect_prompts[aspect] = match.group(1).strip()

    ret_texts, probs = retrieve_text(frames, list(aspect_prompts.values()), models=models, topk=len(aspect_prompts), device=device)
    aspect_scores = dict(zip(aspect_prompts.keys(), probs))
    return aspect_scores

# Video-Text retrieval function
def retrieve_text(frames, texts, models, topk=5, device=device):
    clip, tokenizer = models['viclip'], models['tokenizer']
    
    frames_tensor = frames2tensor(frames, device=device)
    vid_feat = get_vid_feat(frames_tensor, clip)
    
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, dim=0)
    
    probs, idxs = clip.get_predict_label(vid_feat, text_feats_tensor, top=topk)
    ret_texts = [texts[i] for i in idxs.numpy()[0].tolist()]
    return ret_texts, probs.numpy()[0]

# Frame-to-frame similarity calculation
def calculate_frame_to_frame_similarity(frames, models, device=device):
    frame_features = []
    clip = models['viclip']
    
    for frame in frames:
        frame_tensor = frames2tensor([frame], fnum=1, device=device)
        with torch.no_grad():
            frame_feature = get_vid_feat(frame_tensor, clip)
            frame_features.append(frame_feature)
    
    frame_to_frame_similarities = []
    for i in range(1, len(frame_features)):
        similarity = torch.cosine_similarity(frame_features[i - 1], frame_features[i]).item()
        frame_to_frame_similarities.append(similarity)
    
    return np.mean(frame_to_frame_similarities) if frame_to_frame_similarities else 0

# Calculate similarity between overall prompt and video
def calculate_overall_similarity(frames, overall_prompt, models, device=device):
    ret_texts, probs = retrieve_text(frames, [overall_prompt], models=models, topk=1, device=device)
    return ret_texts, probs[0]

# Load JSON data
json_path = "/storage/home/mingzhe/code/VideoGen-of-Thought/data/data_mary_life/image_prompt_pairs.json"
with open(json_path, 'r') as f:
    data = json.load(f)

# Aspects for prompt matching
aspects = ["Character", "Background", "Relation", "Camera Pose", "HDR Description"]

# Initialize a dictionary to accumulate scores for each aspect
aspect_totals = {aspect: 0 for aspect in aspects}

# Calculate similarity scores for each item in JSON data
results = []
for i, item in enumerate(data):
    video_path = f"/storage/home/mingzhe/code/EasyAnimate/result/{str(i+1).zfill(8)}.mp4"
    prompt = item['prompt']

    if not os.path.isfile(video_path):
        print(f"Video file not found: {video_path}")
        continue

    video = cv2.VideoCapture(video_path)
    frames = [x for x in _frame_from_video(video)]
    video.release()

    # Calculate aspect-based similarity
    aspect_scores = calculate_aspect_similarity(prompt, aspects, frames, models=models, device=device)
    for aspect, score in aspect_scores.items():
        aspect_totals[aspect] += score  # Accumulate scores for each aspect
    print(f"Aspect scores for video {i+1}: {aspect_scores}")

    # Calculate overall prompt similarity
    overall_ret_texts, overall_similarity = calculate_overall_similarity(frames, prompt, models=models, device=device)
    print(f"Overall similarity score for video {i+1}: {overall_similarity}")

    # Calculate frame-to-frame similarity
    frame2frame_similarity = calculate_frame_to_frame_similarity(frames, models=models, device=device)
    print(f"Frame-to-frame similarity score for video {i+1}: {frame2frame_similarity}")

    # Store results
    results.append({
        'video_path': video_path,
        'aspect_scores': aspect_scores,
        'overall_similarity_score': overall_similarity,
        'frame_similarity_score': frame2frame_similarity
    })

# Calculate average scores for each aspect across all videos
num_samples = len(results)
aspect_averages = {aspect: total / num_samples for aspect, total in aspect_totals.items()}

# Save individual results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('./result/easyanimate.csv', index=False)

# Display the average scores for each aspect
print("Average similarity scores for each aspect:")
for aspect, avg_score in aspect_averages.items():
    print(f"{aspect}: {avg_score}")
