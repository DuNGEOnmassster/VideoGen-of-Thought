import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import csv
import logging

# Ensure the result directory exists
os.makedirs("./result", exist_ok=True)

# Set up logging
logging.basicConfig(filename="./result/easyanimate_video_similarity_log.txt", level=logging.INFO, format="%(message)s")

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
    return frames

def extract_features(frames, model):
    """Extract features for each frame using insightface model."""
    features = []
    for frame in frames:
        faces = model.get(frame)
        if len(faces) > 0:
            # Assume the first face found is the target for simplicity
            embedding = faces[0].normed_embedding
            # Replace any NaN in the embedding with 0
            embedding = np.nan_to_num(embedding, nan=0.0)
            features.append(embedding)
        else:
            # If no face is detected, append a zero vector for consistency
            features.append(np.zeros(512))
    return features

def compute_frame_similarities(features):
    """Compute cosine similarity between consecutive frames based on features."""
    similarities = []
    for i in range(len(features) - 1):
        # Check if both features are non-zero vectors
        if np.linalg.norm(features[i]) > 0 and np.linalg.norm(features[i + 1]) > 0:
            similarity = 1 - cosine(features[i], features[i + 1])
            # Replace NaN with 0 if cosine similarity calculation results in NaN
            if np.isnan(similarity):
                similarity = 0
        else:
            # If either vector is zero, set similarity to 0
            similarity = 0
        similarities.append(similarity)
    return similarities

def evaluate_video_similarity(video_path):
    """Evaluate frame-to-frame similarity in a video using insightface."""
    # Initialize the FaceAnalysis model
    model = FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(512, 512))

    # Load frames from video
    frames = load_video_frames(video_path)
    logging.info(f"Extracted {len(frames)} frames from {video_path}.")

    # Extract features from each frame
    features = extract_features(frames, model)
    logging.info(f"Feature extraction completed for {video_path}.")

    # Compute frame-to-frame similarities
    similarities = compute_frame_similarities(features)
    logging.info(f"Frame-to-frame similarity evaluation completed for {video_path}.")

    # Calculate the average similarity
    avg_similarity = np.mean(similarities) if similarities else 0
    logging.info(f"Average similarity for {video_path}: {avg_similarity:.4f}")

    return similarities, avg_similarity

def evaluate_all_videos_in_directory(directory):
    """Evaluate all videos in a specified directory."""
    results = {}
    avg_similarities = []  # List to store average similarity for each video

    for i in range(1, 31):  # Assuming videos are named sequentially from 1 to 30
        video_path = os.path.join(directory, f"{i:08d}.mp4")  # Format file names as 0001.mp4, 0002.mp4, etc.
        if os.path.exists(video_path):
            logging.info(f"\nProcessing video: {video_path}")
            similarities, avg_similarity = evaluate_video_similarity(video_path)
            results[video_path] = {"similarities": similarities, "avg_similarity": avg_similarity}
            avg_similarities.append((video_path, avg_similarity))
        else:
            logging.info(f"Video not found: {video_path}")

    # Save average similarities to CSV
    with open("./result/easyanimate_similarities.csv", mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Video Path", "Average Similarity"])
        writer.writerows(avg_similarities)

    return results

# Set the directory path where videos are stored
directory = "/storage/home/mingzhe/code/EasyAnimate/result"
results = evaluate_all_videos_in_directory(directory)

# Print results to log file
for video_path, data in results.items():
    similarities = data["similarities"]
    avg_similarity = data["avg_similarity"]
    logging.info(f"\nSimilarity scores for {video_path}:")
    for i, similarity in enumerate(similarities):
        logging.info(f"  Frame {i} to Frame {i+1}: {similarity:.4f}")
    logging.info(f"Average similarity: {avg_similarity:.4f}")
