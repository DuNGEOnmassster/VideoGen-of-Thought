import torch
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from raft import RAFT
import argparse
import torchvision.transforms as T
import onnxruntime as ort

# RAFT Initialization using the full model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = argparse.Namespace(
    small=False,  # Use the full RAFT model to match the weights
    corr_levels=4,
    corr_radius=4,
    dropout=0,
    alternate_corr=False,
    mixed_precision=False
)

raft = RAFT(args).to(device)  # Full precision for compatibility
raft = torch.nn.DataParallel(raft)
raft.load_state_dict(torch.load('models/raft-sintel.pth'))

# ONNX model paths
detection_model_path = "/storage/home/mingzhe/.insightface/models/antelopev2/antelopev2/scrfd_10g_bnkps.onnx"
recognition_model_path = "/storage/home/mingzhe/.insightface/models/antelopev2/antelopev2/glintr100.onnx"

# Load detection and recognition models with ONNX Runtime
detection_model = ort.InferenceSession(detection_model_path)
recognition_model = ort.InferenceSession(recognition_model_path)

# Preocess the video tensor by adjusting the shape of the generated and real videos to be the same.
def preprocess_video_tensor(videos, target_shape=(32, 3, 240, 320)):
    processed_videos = []
    for video in videos:
        _, C, H, W = video.shape
        T_target, C_target, H_target, W_target = target_shape

        # Adjust the number of frames
        if video.shape[0] != T_target:
            video = video[:T_target] if video.shape[0] > T_target else torch.cat([video, video[-1:].repeat(T_target - video.shape[0], 1, 1, 1)], dim=0)

        # Adjust the height and width
        video = torchvision.transforms.functional.resize(video, (H_target, W_target))

        processed_videos.append(video)
    return torch.stack(processed_videos)


def detect_faces(frame):
    """
    Perform face detection and get face embeddings.
    """
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/127.5, size=(640, 640), mean=(127.5, 127.5, 127.5))
    detection_output = detection_model.run(None, {"input.1": blob})[0]
    
    # Debugging: Print detection_output to inspect its structure
    print("Detection output:", detection_output)
    print("Detection output shape:", detection_output.shape)
    
    faces = []
    for det in detection_output:
        # Ensure det has at least 5 elements before accessing index 4
        if len(det) >= 5 and det[4] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, det[:4])
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                embedding = get_face_embedding(face)
                faces.append(embedding)

    return faces[0] if faces else None

def get_face_embedding(face):
    """
    Get the face embedding for a detected face using the recognition model.
    """
    face_resized = cv2.resize(face, (112, 112))
    face_blob = cv2.dnn.blobFromImage(face_resized, scalefactor=1.0 / 255)
    embedding = recognition_model.run(None, {"data": face_blob})[0]
    return embedding.flatten()

def extract_flow(frame1, frame2, device):
    """
    Use RAFT to compute optical flow between two frames.
    """
    transform = T.Compose([T.ToTensor()])
    frame1_tensor = transform(frame1).unsqueeze(0).to(device)
    frame2_tensor = transform(frame2).unsqueeze(0).to(device)

    with torch.no_grad():
        flow_predictions = raft(frame1_tensor, frame2_tensor)
        flow = flow_predictions[-1]
    flow_magnitude = flow.norm(dim=1)
    mean_flow_magnitude = flow_magnitude.mean().item()
    return mean_flow_magnitude

def process_video_frame_pair(frame1, frame2, device):
    """
    Process two frames, compute optical flow and face similarity.
    Return (smoothness score, face similarity score).
    """
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # Compute optical flow smoothness score
    flow_magnitude = extract_flow(frame1_rgb, frame2_rgb, device)
    flow_smoothness_score = 1 / (1 + flow_magnitude)

    # Compute face similarity score
    face_feature1 = detect_faces(frame1_rgb)
    face_feature2 = detect_faces(frame2_rgb)
    face_similarity_score = 0

    if face_feature1 is not None and face_feature2 is not None:
        face_similarity_score = 1 - cosine(face_feature1, face_feature2)

    # Clear CUDA cache to free memory after each frame processing
    torch.cuda.empty_cache()
    
    return flow_smoothness_score, face_similarity_score

def evaluate_video(video_path):
    """
    Process video frames sequentially to reduce memory usage.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    flow_scores = []
    face_similarity_scores = []

    # Process frames sequentially instead of in parallel to reduce memory usage
    for i in range(len(frames) - 1):
        flow_score, face_similarity_score = process_video_frame_pair(frames[i], frames[i + 1], device)
        flow_scores.append(flow_score)
        face_similarity_scores.append(face_similarity_score)

    avg_flow_score = np.mean(flow_scores) if flow_scores else 0
    avg_face_similarity_score = np.mean(face_similarity_scores) if face_similarity_scores else 0

    return avg_flow_score, avg_face_similarity_score

# Example usage
video_path = '../samples/1.mp4'
avg_flow_score, avg_face_similarity_score = evaluate_video(video_path)

print(f"Average Flow Smoothness Score: {avg_flow_score}")
print(f"Average Face Similarity Score: {avg_face_similarity_score}")
