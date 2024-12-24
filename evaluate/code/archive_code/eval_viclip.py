import os
import pandas as pd
import torch
import cv2
import numpy as np
from transformers import AutoModel, AutoTokenizer
import re
import json

# Utility function to iterate over frames from a video
def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

def safely_parse_json(json_str):
    # Replace single quotes with double quotes to convert to valid JSON format
    json_str = re.sub(r"(?<!\\)'", '"', json_str)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"text:{json_str}")
        print(f"Error parsing JSON: {e}")
        return None

# Load ViCLIP model from local directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local ViCLIP directory path (update this to your actual path)
viclip_dir = "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/ViCLIP_B_16"

# Eval Data Path
data_path = "/storage/home/mingzhe/dataset/vgot_data/videos/"

# Load the model and tokenizer from local path
model = AutoModel.from_pretrained(viclip_dir, trust_remote_code=True).to(device)
tokenizer = model.tokenizer
models = {"viclip": model, "tokenizer": tokenizer}
model.eval()  # Set model to evaluation mode

# Define normalization and video processing utilities
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

def normalize(data):
    return (data / 255.0 - v_mean) / v_std

def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    # Use all available frames if they are fewer than `fnum`
    actual_fnum = min(len(vid_list), fnum)

    # Step size to sample frames evenly
    step = max(1, len(vid_list) // actual_fnum)
    selected_frames = vid_list[::step][:actual_fnum]

    # Resize and normalize frames
    resized_frames = [cv2.resize(x[:, :, ::-1], target_size) for x in selected_frames]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in resized_frames]
    
    # Stack frames to create a tensor
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()

    return vid_tube

# Function to extract video-level features using ViCLIP
def get_vid_feat(frames_tensor, clip):
    return clip.get_vid_features(frames_tensor)

# Function to get text features using ViCLIP and tokenizer
def get_text_feat_dict(texts, clip, tokenizer, text_feat_d={}):
    for t in texts:
        with torch.no_grad():
            tokens = tokenizer.encode(t)
            tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
            feat = clip.get_text_features(tokens_tensor, tokenizer)
            text_feat_d[t] = feat
    return text_feat_d

# Video-Text retrieval function
def retrieve_text(frames, texts, models, topk=5, device=torch.device('cuda')):
    assert isinstance(models, dict) and models['viclip'] is not None and models['tokenizer'] is not None
    clip, tokenizer = models['viclip'], models['tokenizer']
    clip = clip.to(device)
    
    # Convert frames to tensor representation
    frames_tensor = frames2tensor(frames, device=device)
    
    # Get video features
    vid_feat = get_vid_feat(frames_tensor, clip)
    
    # Get text features for the given texts
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, dim=0)
    
    # Predict label (i.e., calculate similarity and retrieve the most relevant texts)
    probs, idxs = clip.get_predict_label(vid_feat, text_feats_tensor, top=topk)

    ret_texts = [texts[i] for i in idxs.numpy()[0].tolist()]
    return ret_texts, probs.numpy()[0]

# Function to calculate average frame-to-frame similarity
def calculate_frame_to_frame_similarity(frames, models, device=torch.device('cuda')):
    frame_features = []
    clip = models['viclip']
    
    # Extract frame-level features
    for frame in frames:
        frame_tensor = frames2tensor([frame], fnum=1, device=device)
        with torch.no_grad():
            frame_feature = get_vid_feat(frame_tensor, clip)
            frame_features.append(frame_feature)
    
    # Calculate similarities between consecutive frames
    frame_to_frame_similarities = []
    for i in range(1, len(frame_features)):
        similarity = torch.cosine_similarity(frame_features[i - 1], frame_features[i]).item()
        frame_to_frame_similarities.append(similarity)
    
    # Return average similarity
    return np.mean(frame_to_frame_similarities) if frame_to_frame_similarities else 0

# Function to calculate similarity between overall prompt and video
def calculate_overall_similarity(frames, overall_prompt, models, device=torch.device('cuda')):
    # Use retrieve_text to calculate similarity
    ret_texts, probs = retrieve_text(frames, [overall_prompt], models=models, topk=1, device=device)
    return ret_texts, probs[0]

# Read the CSV file to get page_dir, videoid, and prompts
csv_path = '/storage/home/mingzhe/dataset/vgot_data/VGoT_dataset.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_path)

# Iterate through each row in the CSV and calculate similarity scores
results = []

for _, row in df.iterrows():
    page_dir = row['page_dir']
    videoid = row['videoid']
    
    # Convert the `name` column to a dictionary (since it's a string representation of a dictionary)
    json_str = row['name']
    prompts_dict = safely_parse_json(json_str)
    if prompts_dict is None:
        continue  # Skip rows that failed parsing
    
    import pdb; pdb.set_trace()
    # Extract the prompts using the keys in the dictionary
    prompts = [prompts_dict[key] for key in ['character', 'Background', 'Relation', 'Camera Pose', 'HDR Description']]

    # Form the video path
    page_dir = os.path.join(data_path, page_dir)
    video_path = os.path.join(page_dir, videoid + ".mp4")

    # Load video frames
    video = cv2.VideoCapture(video_path)
    frames = [x for x in _frame_from_video(video)]
    video.release()

    # Retrieve texts based on video content
    ret_texts, probs = retrieve_text(frames, prompts, models=models, topk=5, device=device)
    print(f"text: {ret_texts}")
    print(f"5 aspect scores:{probs}")

    # Overall prompt similarity calculation
    overall_prompt = " ".join(prompts)
    overall_ret_texts, overall_similarity = calculate_overall_similarity(frames, overall_prompt, models=models, device=device)
    print(f"text: {overall_ret_texts}")
    print(f"overall similarity score: {overall_similarity}")

    # Frame-to-frame similarity calculation
    frame2frame_similarity = calculate_frame_to_frame_similarity(frames, models=models, device=device)
    print(f"frame similarity score: {frame2frame_similarity}")

    # Store the results
    results.append({
        'videoid': videoid,
        'retrieved_texts': ret_texts,
        'probs': probs.tolist(),
        'overall_similarity_score': overall_similarity,
        'frame_similarity_score': frame2frame_similarity
    })

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('./retrieval_similarity_scores.csv', index=False)

print("Similarity scores calculated and saved successfully.")