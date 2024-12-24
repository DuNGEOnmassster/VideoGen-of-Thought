import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import csv
import logging

# Load and preprocess frames for Face Analysis
def load_video_frames(video_path, frame_size=(512, 512)):
    """Load frames from the video and resize to the given frame size."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames

def extract_face_features(frames, model):
    """Extract face features for each frame using insightface model."""
    features = []
    for idx, frame in enumerate(frames):
        faces = model.get(frame)
        if len(faces) > 0:
            embedding = faces[0].normed_embedding
            embedding = np.nan_to_num(embedding, nan=0.0)
            features.append(embedding)
        else:
            features.append(np.zeros(512))
        print(f"Frame {idx + 1}/{len(frames)}: Feature extracted.")
    return features

def compute_frame_similarities(features):
    """Compute cosine similarity between consecutive frames based on features."""
    similarities = []
    for i in range(len(features) - 1):
        similarity = 1 - cosine(features[i], features[i + 1])
        if np.isnan(similarity):
            similarity = 0
        similarities.append(similarity)
    avg_similarity = np.mean(similarities) if similarities else 0
    print(f"Average within-video face similarity: {avg_similarity:.4f}")
    return avg_similarity

def compute_cross_shot_similarity(prev_end_features, next_start_features):
    """Compute average similarity between the last 8 frames of one video and the first 8 frames of the next video."""
    cross_shot_similarities = []
    for i in range(len(prev_end_features)):
        similarity = 1 - cosine(prev_end_features[i], next_start_features[i])
        if np.isnan(similarity):
            similarity = 0
        cross_shot_similarities.append(similarity)
    avg_cross_shot_similarity = np.mean(cross_shot_similarities) if cross_shot_similarities else 0
    print(f"Cross-shot face similarity: {avg_cross_shot_similarity:.4f}")
    return avg_cross_shot_similarity

def evaluate_video_face_similarity(video_path, model):
    """Evaluate frame-to-frame face similarity in a video using insightface."""
    frames = load_video_frames(video_path)
    features = extract_face_features(frames, model)
    avg_frame_similarity = compute_frame_similarities(features)
    return features, avg_frame_similarity

def evaluate_story_directory(story_path, result_dir):
    """Evaluate cross-shot and within-video face similarities for all videos in a specified story directory."""
    # Check if story path exists
    if not os.path.exists(story_path):
        print(f"Story directory not found: {story_path}")
        return None, None
    
    log_file_path = os.path.join(result_dir, "face_similarity_log.txt")
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(message)s")

    model = FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(512, 512))

    start_features = []
    end_features = []
    all_cross_shot_similarities = []
    all_within_video_similarities = []

    # Adjust for different naming conventions in different directories
    video_files = sorted([f for f in os.listdir(story_path) if f.endswith(".mp4")])

    for idx, video_name in enumerate(video_files):
        video_path = os.path.join(story_path, video_name)
        if os.path.exists(video_path):
            print(f"Processing video: {video_path}")
            logging.info(f"\nProcessing video: {video_path}")
            features, avg_frame_similarity = evaluate_video_face_similarity(video_path, model)
            all_within_video_similarities.append(avg_frame_similarity)
            if len(features) >= 8:
                if idx == 0:
                    end_features.append(features[-8:])
                elif idx == len(video_files) - 1:
                    start_features.append(features[:8])
                else:
                    start_features.append(features[:8])
                    end_features.append(features[-8:])
        else:
            print(f"Video not found: {video_path}")
            logging.info(f"Video not found: {video_path}")

    for i in range(len(end_features)):
        if i + 1 < len(start_features):
            avg_similarity = compute_cross_shot_similarity(end_features[i], start_features[i + 1])
            all_cross_shot_similarities.append(avg_similarity)
            print(f"Cross-shot face similarity between video {i+1} and video {i+2}: {avg_similarity:.4f}")
            logging.info(f"Cross-shot face similarity between video {i+1} and video {i+2}: {avg_similarity:.4f}")

    overall_avg_cross_shot_similarity = np.mean(all_cross_shot_similarities) if all_cross_shot_similarities else 0
    overall_avg_within_video_similarity = np.mean(all_within_video_similarities) if all_within_video_similarities else 0
    print(f"\nOverall average cross-shot face similarity: {overall_avg_cross_shot_similarity:.4f}")
    print(f"Overall average within-video face similarity: {overall_avg_within_video_similarity:.4f}")
    logging.info(f"\nOverall average cross-shot face similarity: {overall_avg_cross_shot_similarity:.4f}")
    logging.info(f"Overall average within-video face similarity: {overall_avg_within_video_similarity:.4f}")

    # Save results to CSV
    with open(os.path.join(result_dir, "face_overall_similarities.csv"), mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Story", "Avg Cross-shot Face Similarity", "Avg Within-video Face Similarity"])
        writer.writerow([os.path.basename(story_path), overall_avg_cross_shot_similarity, overall_avg_within_video_similarity])

    return overall_avg_cross_shot_similarity, overall_avg_within_video_similarity

# Define base directories for each dataset
base_directory = "/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/results/Ablation"
sub_directories = [
    "with_EP_with_IP/samples_separate",
    "with_EP_without_IP/samples_separate",
    "without_EP_with_IP/samples_separate",
    "without_EP_without_IP/samples_separate"
]

# Run evaluation on all subdirectories
for sub_dir in sub_directories:
    story_path = os.path.join(base_directory, sub_dir)
    result_dir = story_path
    evaluate_story_directory(story_path, result_dir)