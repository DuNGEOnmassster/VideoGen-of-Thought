import os
import json
import torch
import cv2
import numpy as np
import pandas as pd
from transformers import AutoModel
import math
from ignite.metrics import PSNR
from torchvision.models import inception_v3
from torch.nn import functional as F
from scipy.stats import entropy
from torchvision import transforms

# Utility function to iterate over frames from a video
def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

# Load ViCLIP model from local directory
device = torch.device("cuda")
viclip_dir = "/data/nas/mingzhe/code-release/VideoGen-of-Thought-official/weights/ViCLIP-B-16-hf"
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

def get_text_feat(text, clip, tokenizer):
    with torch.no_grad():
        tokens = tokenizer.encode(text)
        tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        return clip.get_text_features(tokens_tensor, tokenizer)

def calculate_similarity(prompt, frames, models, device):
    clip, tokenizer = models['viclip'], models['tokenizer']
    frames_tensor = frames2tensor(frames, device=device)
    vid_feat = get_vid_feat(frames_tensor, clip)
    text_feat = get_text_feat(prompt, clip, tokenizer)
    
    # Compute cosine similarity between video and text features
    similarity = torch.nn.functional.cosine_similarity(vid_feat, text_feat, dim=-1)
    return similarity.item()

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

def calculate_inception_score(frames, model, n_split=10, eps=1e-16):
    """Calculate Inception Score for the frames using a pre-trained InceptionV3 model.
    """
    # If there are no frames or too few frames, return 0
    if len(frames) < 2:
        return 0.0
    
    # Load pre-trained Inception V3 model
    try:
        inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        inception_model.eval()
        
        # Remove the final fully connected layer to get features
        inception_model.fc = torch.nn.Identity()
        
        # Prepare pre-processing transformation
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Process each frame to get prediction results
        preds = []
        with torch.no_grad():
            for frame in frames:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Pre-process the frame
                input_tensor = preprocess(frame_rgb).unsqueeze(0).to(device)
                # Get prediction
                pred = F.softmax(inception_model(input_tensor), dim=1).cpu().numpy()
                preds.append(pred)
        
        # Convert to numpy array
        preds = np.concatenate(preds, axis=0)
        p_y = np.mean(preds, axis=0)
        
        # Calculate the KL divergence for each split
        split_scores = []
        for i in range(n_split):
            # If the number of frames is less than the number of splits, we repeat the frames
            part = preds[i * (len(preds) // max(1, n_split - 1)):(i + 1) * (len(preds) // max(1, n_split - 1)), :]
            if len(part) == 0:
                continue
            
            p_y_given_x = part
            kl_d = p_y_given_x * (np.log(p_y_given_x + eps) - np.log(p_y + eps))
            kl_d = np.mean(np.sum(kl_d, axis=1))
            split_scores.append(np.exp(kl_d))
        
        # Calculate the final Inception Score (mean ± standard deviation)
        is_mean = np.mean(split_scores) if split_scores else 0.0
        is_std = np.std(split_scores) if len(split_scores) > 1 else 0.0
        
        return is_mean
    
    except Exception as e:
        print(f"Error calculating Inception Score: {e}")
        # If an error occurs, return a default value
        return 1.0

# Define paths and settings
json_data_base_dir = "/data/nas/mingzhe/code-release/VideoGen-of-Thought-official/evaluate/Ablation_Studies"
text_prompt_path = os.path.join(json_data_base_dir, "cyclist/test_prompts.txt")
with open(text_prompt_path, 'r') as f:
    data = [line.strip() for line in f.readlines()]

base_directories = {
    "with_EP_with_IP": "/data/nas/mingzhe/code-release/VideoGen-of-Thought-official/Experiment/Ablation/with_EP_with_IP/samples_separate",
    "with_EP_without_IP": "/data/nas/mingzhe/code-release/VideoGen-of-Thought-official/Experiment/Ablation/with_EP_without_IP/samples_separate",
    "without_EP_with_IP": "/data/nas/mingzhe/code-release/VideoGen-of-Thought-official/Experiment/Ablation/without_EP_with_IP/samples_separate",
    "without_EP_without_IP": "/data/nas/mingzhe/code-release/VideoGen-of-Thought-official/Experiment/Ablation/without_EP_without_IP/samples_separate"
}

# Initialize global dictionary to store scores
global_scores = {method: [] for method in base_directories}
psnr_scores = {method: [] for method in base_directories}
inception_scores = {method: [] for method in base_directories}

# Process each story
for method, video_dir in base_directories.items():
    result_dir = os.path.join(video_dir, "t2v_clip_score_result")
    os.makedirs(result_dir, exist_ok=True)

    # Calculate scores for each prompt
    for i, prompt in enumerate(data):
        video_name = f"sample_ip_{i}_sample0.mp4"
        video_path = os.path.join(video_dir, video_name)
        if not os.path.isfile(video_path):
            print(f"Video file not found: {video_path}")
            continue

        video = cv2.VideoCapture(video_path)
        frames = [x for x in _frame_from_video(video)]
        video.release()

        if frames:
            similarity_score = calculate_similarity(prompt, frames, models=models, device=device)
            global_scores[method].append(similarity_score)
            psnr_score = calculate_psnr(frames)
            psnr_scores[method].append(psnr_score)
            inception_score = calculate_inception_score(frames, model)
            inception_scores[method].append(inception_score)

    # Save individual method scores for each story
    scores_df = pd.DataFrame({
        "Clip Score": global_scores[method],
        "PSNR Score": psnr_scores[method],
        "Inception Score": inception_scores[method]
    })
    if not scores_df.empty:
        avg_clip_score = scores_df["Clip Score"].mean()
        avg_psnr_score = scores_df["PSNR Score"].mean()
        avg_inception_score = scores_df["Inception Score"].mean()
        scores_df = scores_df.append({
            "Clip Score": avg_clip_score,
            "PSNR Score": avg_psnr_score,
            "Inception Score": avg_inception_score
        }, ignore_index=True)
        scores_df = scores_df.append({"Clip Score": f"Average Scores - Clip: {avg_clip_score}, PSNR: {avg_psnr_score}, Inception: {avg_inception_score}"}, ignore_index=True)
        scores_df.to_csv(os.path.join(result_dir, f"{method}_clip_scores.csv"), index=False)
        print(f"Saved per-story scores for {method} in {result_dir}")

# Calculate and save overall averages across all methods
overall_averages = {
    method: {
        "Average Clip Score": np.mean(global_scores[method]) if global_scores[method] else 0,
        "Average PSNR Score": np.mean(psnr_scores[method]) if psnr_scores[method] else 0,
        "Average Inception Score": np.mean(inception_scores[method]) if inception_scores[method] else 0
    } for method in base_directories
}

# Save overall average scores
overall_avg_df = pd.DataFrame.from_dict(overall_averages, orient='index')
overall_avg_df.to_csv("/data/nas/mingzhe/code-release/VideoGen-of-Thought-official/Experiment/Ablation_clip_scores.csv")
print("Saved overall average scores across all methods in 'ablation_clip_scores.csv'")