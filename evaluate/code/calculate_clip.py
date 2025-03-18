import torch
import cv2
import json
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, List

# Configure parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 50  # Modify to sample 50 frames
VIDEO_EXT = ".mp4"  # Video file extension

# Initialize the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def load_captions(json_path: str) -> List[str]:
    """Load the JSON file containing captions"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["stories"]


def extract_video_frames(video_path: str) -> List[Image.Image]:
    """Extract 50 frames uniformly from the video (handle short videos)"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    # Calculate the sampling interval (handle the case where the total number of frames is insufficient)
    step = max(total_frames // NUM_FRAMES, 1)
    sampled_frames = 0
    current_frame = 0

    while sampled_frames < NUM_FRAMES and current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            sampled_frames += 1
        current_frame += step

    # If still insufficient, loop back to start to supplement
    while sampled_frames < NUM_FRAMES:
        current_frame %= total_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            sampled_frames += 1
        current_frame += 1

    cap.release()
    return frames[:NUM_FRAMES]  # Ensure to return exactly 50 frames


def process_videos(video_dir: str, caption_data: List[str], output_file: str):
    """Batch process videos and save results"""
    results = []
    total_score = 0

    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(VIDEO_EXT)])

    for i, video_name in enumerate(video_files):
        caption = caption_data[i] if i < len(caption_data) else None
        if not caption:
            print(f"Warning: No caption found for {video_name}")
            continue

        # Process a single video
        video_path = os.path.join(video_dir, video_name)
        try:
            frames = extract_video_frames(video_path)
            # Step 1: Process text and image inputs separately
            text_inputs = processor(
                text=[caption],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(DEVICE)

            image_inputs = processor(
                images=frames,
                return_tensors="pt",
                padding=True
            ).to(DEVICE)

            with torch.no_grad():
                # Step 2: Get image and text features separately
                image_features = model.get_image_features(**image_inputs)
                text_features = model.get_text_features(**text_inputs)

            # Step 3: Correctly calculate cosine similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).mean().item()  # Average similarity across all frames

            total_score += similarity

            results.append({
                "video": video_name,
                "caption": caption,
                "score": similarity
            })
            print(f"Processed {video_name} - Score: {similarity:.4f}")

        except Exception as e:
            print(f"Error processing {video_name}: {str(e)}")

    results.append({"total_score": total_score})

    # Save the results to a JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    # Usage example
    video_directory = "VideoCrafter2/"  # Define the video directory
    caption_json = "captions.json"  # Define the caption file path
    output_txt = "VideoCrafter2_clip_scores.json"  # Define the output file path, modify to JSON file

    # Load the caption data
    captions = load_captions(caption_json)

    # Process all videos
    process_videos(
        video_dir=video_directory,
        caption_data=captions,
        output_file=output_txt
    )