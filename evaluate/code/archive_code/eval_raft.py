import insightface
import torch
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cosine
from raft import RAFT
import argparse
import torchvision.transforms as T

# RAFT Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = argparse.Namespace(
    small=False,  # Use small model (set True if you want to use the small version)
    corr_levels=4,  # Number of correlation levels
    corr_radius=4,  # Radius for correlation
    dropout=0,  # Dropout rate
    alternate_corr=False,  # Whether to use alternate correlation method
    mixed_precision=False  # Whether to use mixed precision
)

raft = RAFT(args).to(device)  # Initialize RAFT with args
raft = torch.nn.DataParallel(raft)  # Wrap RAFT with DataParallel for multiple GPUs
raft.load_state_dict(torch.load('models/raft-sintel.pth'))  # Load pre-trained weights

def extract_flow(frame1, frame2, device):
    """
    Use RAFT to compute optical flow between two frames
    """
    # Convert frames to PyTorch tensors in [C, H, W] format and normalize to [0, 1]
    transform = T.Compose([
        T.ToTensor(),  # Converts a NumPy array in [H, W, C] to a tensor in [C, H, W] and normalizes to [0, 1]
    ])
    
    frame1_tensor = transform(frame1).unsqueeze(0).to(device)  # Add batch dimension -> [1, C, H, W]
    frame2_tensor = transform(frame2).unsqueeze(0).to(device)  # Add batch dimension -> [1, C, H, W]

    with torch.no_grad():
        flow_predictions = raft(frame1_tensor, frame2_tensor)  # Output is a list of flow predictions
        flow = flow_predictions[-1]  # Use the final flow prediction

    # Calculate the flow magnitude (movement between frames)
    flow_magnitude = flow.norm(dim=1)  # Magnitude of the flow across the channels
    mean_flow_magnitude = flow_magnitude.mean().item()  # Average flow magnitude

    return mean_flow_magnitude

def process_video_frame_pair(frame1, frame2, device):
    """
    Process two frames, compute optical flow and face similarity
    Return (smoothness score, face similarity score)
    """
    # Convert frames to RGB format
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # 1. Compute optical flow smoothness score
    flow_magnitude = extract_flow(frame1_rgb, frame2_rgb, device)
    flow_smoothness_score = 1 / (1 + flow_magnitude)  # Lower flow magnitude means smoother

    return flow_smoothness_score

def evaluate_video(video_path):
    """
    Process video frames in parallel, calculate flow and face consistency
    """
    # Read the video
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # Store each frame

    cap.release()

    # Use ThreadPoolExecutor for parallel processing of consecutive frames
    flow_scores = []

    with ThreadPoolExecutor() as executor:
        futures = []
        num_gpus = torch.cuda.device_count()
        for i in range(len(frames) - 1):
            gpu_id = i % num_gpus  # Distribute tasks across available GPUs
            device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
            futures.append(executor.submit(process_video_frame_pair, frames[i], frames[i+1], device))

        for future in futures:
            flow_score = future.result()
            flow_scores.append(flow_score)

    # Calculate average flow smoothness
    avg_flow_score = np.mean(flow_scores) if flow_scores else 0

    return avg_flow_score

# Example usage
video_path = '../samples/1.mp4'  # Path to your video file
avg_flow_score = evaluate_video(video_path)

print(f"Average Flow Smoothness Score: {avg_flow_score}")
