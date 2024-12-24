import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.spatial.distance import cosine
import csv
import logging

# Define different naming formats for each dataset
NAMING_FORMATS = {
    "VGoT": "sample_ip_{:d}_sample0.mp4",
    "CogVideo": "video_{:02d}.mp4",
    "VideoCrafter1": "{:04d}.mp4",
    "VideoCrafter2": "{:04d}.mp4",
    "EasyAnimate": "{:08d}.mp4"
}

# Pre-trained VGG19 model for style feature extraction
class StyleFeatureExtractor(nn.Module):
    def __init__(self):
        super(StyleFeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features
        self.model = nn.Sequential(*list(vgg19.children())[:16])  # Up to block5_conv4
    
    def forward(self, x):
        return self.model(x)

# Load and preprocess frames for VGG19
def preprocess_frame(frame, frame_size=(224, 224)):
    """Preprocess the frame for VGG19 input."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(frame_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0)

def load_video_frames(video_path, frame_size=(224, 224)):
    """Load frames from the video and resize to the given frame size."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = preprocess_frame(frame, frame_size)
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames

def extract_style_features(frames, model):
    """Extract style features for each frame using VGG19 model."""
    features = []
    for idx, frame in enumerate(frames):
        with torch.no_grad():
            frame = frame.cuda()
            feature = model(frame)  # Extract features
            features.append(feature.squeeze().cpu().numpy())
        print(f"Frame {idx + 1}/{len(frames)}: Feature extracted.")
    return features

def compute_frame_style_similarities(features):
    """Compute cosine similarity between consecutive frames based on style features."""
    similarities = []
    for i in range(len(features) - 1):
        # Flatten the 2D feature tensors into 1D
        flattened_feature_i = features[i].flatten()
        flattened_feature_next = features[i + 1].flatten()

        similarity = 1 - cosine(flattened_feature_i, flattened_feature_next)
        if np.isnan(similarity):
            similarity = 0
        similarities.append(similarity)
    avg_similarity = np.mean(similarities) if similarities else 0
    print(f"Average within-video style similarity: {avg_similarity:.4f}")
    return avg_similarity

def compute_cross_shot_style_similarity(prev_end_features, next_start_features):
    """Compute average style similarity between the last 8 frames of one video and the first 8 frames of the next video."""
    cross_shot_similarities = []
    for i in range(len(prev_end_features)):
        # Flatten the 2D feature tensors into 1D
        flattened_prev_end_feature = prev_end_features[i].flatten()
        flattened_next_start_feature = next_start_features[i].flatten()

        similarity = 1 - cosine(flattened_prev_end_feature, flattened_next_start_feature)
        if np.isnan(similarity):
            similarity = 0
        cross_shot_similarities.append(similarity)
    avg_cross_shot_similarity = np.mean(cross_shot_similarities) if cross_shot_similarities else 0
    print(f"Cross-shot style similarity: {avg_cross_shot_similarity:.4f}")
    return avg_cross_shot_similarity

def evaluate_video_style_similarity(video_path, model):
    """Evaluate frame-to-frame style similarity in a video using VGG19."""
    frames = load_video_frames(video_path)
    features = extract_style_features(frames, model)
    avg_frame_similarity = compute_frame_style_similarities(features)
    return features, avg_frame_similarity

def evaluate_story_directory(story_path, story_name, naming_format, result_dir, is_vgot=False):
    """Evaluate cross-shot and within-video style similarities for all videos in a specified story directory."""
    if is_vgot:
        story_path = os.path.join(story_path, "samples_separate")
    
    # Check if story path exists
    if not os.path.exists(story_path):
        print(f"Story directory not found: {story_path}")
        return None, None
    
    log_file_path = os.path.join(result_dir, f"{story_name}_style_similarity_log.txt")
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(message)s")

    model = StyleFeatureExtractor()
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    start_features = []
    end_features = []
    all_cross_shot_similarities = []
    all_within_video_similarities = []

    for i in range(1, 31):  # Assuming 30 shot-videos per story
        video_name = naming_format.format(i)
        video_path = os.path.join(story_path, video_name)
        if os.path.exists(video_path):
            print(f"Processing video: {video_path}")
            logging.info(f"\nProcessing video: {video_path}")
            features, avg_frame_similarity = evaluate_video_style_similarity(video_path, model)
            all_within_video_similarities.append(avg_frame_similarity)
            if len(features) >= 8:
                if i == 1:
                    end_features.append(features[-8:])
                elif i == 30:
                    start_features.append(features[:8])
                else:
                    start_features.append(features[:8])
                    end_features.append(features[-8:])
        else:
            print(f"Video not found: {video_path}")
            logging.info(f"Video not found: {video_path}")

    for i in range(len(end_features)):
        if i + 1 < len(start_features):
            avg_similarity = compute_cross_shot_style_similarity(end_features[i], start_features[i + 1])
            all_cross_shot_similarities.append(avg_similarity)
            print(f"Cross-shot style similarity between video {i+1} and video {i+2}: {avg_similarity:.4f}")
            logging.info(f"Cross-shot style similarity between video {i+1} and video {i+2}: {avg_similarity:.4f}")

    overall_avg_cross_shot_similarity = np.mean(all_cross_shot_similarities) if all_cross_shot_similarities else 0
    overall_avg_within_video_similarity = np.mean(all_within_video_similarities) if all_within_video_similarities else 0
    print(f"\nOverall average cross-shot style similarity for {story_name}: {overall_avg_cross_shot_similarity:.4f}")
    print(f"Overall average within-video style similarity for {story_name}: {overall_avg_within_video_similarity:.4f}")
    logging.info(f"\nOverall average cross-shot style similarity for {story_name}: {overall_avg_cross_shot_similarity:.4f}")
    logging.info(f"Overall average within-video style similarity for {story_name}: {overall_avg_within_video_similarity:.4f}")

    # Save results to CSV
    with open(os.path.join(result_dir, f"{story_name}_style_similarities.csv"), mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Story", "Avg Cross-shot Style Similarity", "Avg Within-video Style Similarity"])
        writer.writerow([story_name, overall_avg_cross_shot_similarity, overall_avg_within_video_similarity])

    return overall_avg_cross_shot_similarity, overall_avg_within_video_similarity

# The rest of the code for `evaluate_all_directories` remains the same as before.
def evaluate_all_directories(base_directories):
    """Evaluate similarities for each story directory in multiple base directories."""
    for base_name, base_directory in base_directories.items():
        # Create a single result directory for each method at the top level
        result_dir = os.path.join(base_directory, "result")
        os.makedirs(result_dir, exist_ok=True)

        overall_cross_shot_similarities = []
        overall_within_video_similarities = []
        
        naming_format = NAMING_FORMATS[base_name]
        is_vgot = base_name == "VGoT"

        for story_name in os.listdir(base_directory):
            story_path = os.path.join(base_directory, story_name)
            if os.path.isdir(story_path):
                print(f"Evaluating story: {story_name} in {base_name}")
                avg_cross_shot_similarity, avg_within_video_similarity = evaluate_story_directory(
                    story_path, story_name, naming_format, result_dir, is_vgot
                )
                
                # Check for None values to handle missing directories
                if avg_cross_shot_similarity is not None and avg_within_video_similarity is not None:
                    overall_cross_shot_similarities.append(avg_cross_shot_similarity)
                    overall_within_video_similarities.append(avg_within_video_similarity)
            else:
                print(f"Story directory not found: {story_path}")

        # Save the overall average for the current dataset in the result directory
        overall_results_path = os.path.join(result_dir, f"{base_name}_style_overall_similarities.csv")
        with open(overall_results_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Metric", "Average"])
            writer.writerow(["Overall Avg Cross-shot Similarity", np.mean(overall_cross_shot_similarities)])
            writer.writerow(["Overall Avg Within-video Similarity", np.mean(overall_within_video_similarities)])
        print(f"Saved overall results for {base_name} to {overall_results_path}")

# Define base directories for each dataset
base_directories = {
    "VGoT": "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/results/VGoT",
    "CogVideo": "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/results/CogVideo",
    "VideoCrafter1": "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/results/VideoCrafter1",
    "VideoCrafter2": "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/results/VideoCrafter2",
    "EasyAnimate": "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/results/EasyAnimate"
}

# Run evaluation on all base directories
evaluate_all_directories(base_directories)
