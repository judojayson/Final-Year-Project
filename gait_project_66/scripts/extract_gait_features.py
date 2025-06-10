import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Paths
VIDEO_ROOT = Path("videos")
OUTPUT_ROOT = Path("data/raw")

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose

def extract_features(video_path, save_path, label):
    """
    Extracts x, y coordinates from 33 pose landmarks for each frame of a video,
    and saves them with label to a CSV file for dataset creation.
    """
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            row = []
            for lm in results.pose_landmarks.landmark[:33]:
                row.extend([lm.x, lm.y])
            row.append(label)
            frames.append(row)

    cap.release()
    pose.close()

    if frames:
        os.makedirs(save_path, exist_ok=True)
        df = pd.DataFrame(frames)
        df.to_csv(save_path / f"{video_path.stem}.csv", index=False, header=False)

def extract_features_from_video(video_path):
    """
    Extracts x, y coordinates of 33 pose landmarks from all valid frames in a video.
    Returns the mean feature vector (66 values) for prediction.
    """
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(str(video_path))
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            row = []
            for lm in results.pose_landmarks.landmark[:33]:
                row.extend([lm.x, lm.y])
            features.append(row)

    cap.release()
    pose.close()

    if features:
        return np.mean(features, axis=0)  # Return average of all frames
    else:
        return None  # No pose detected

def process_all():
    """
    Iterates through videos in 'videos/normal' and 'videos/cerebellar',
    and extracts + saves features for all of them into 'data/raw'.
    """
    for label_name, label_id in [("normal", 0), ("cerebellar", 1)]:
        video_dir = VIDEO_ROOT / label_name
        save_dir = OUTPUT_ROOT / label_name
        for file in tqdm(video_dir.glob("*.mp4"), desc=f"Processing {label_name}"):
            extract_features(file, save_dir, label_id)

if __name__ == "__main__":
    process_all()
