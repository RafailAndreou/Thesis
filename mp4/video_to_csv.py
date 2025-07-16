# video_to_csv.py (updated)
import cv2
import csv
import numpy as np
import mediapipe as mp
from pathlib import Path

def extract_skeleton_from_video(video_path):
    video_path = Path(video_path)
    output_csv_path = video_path.with_suffix(".skeleton.csv")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

    cap = cv2.VideoCapture(str(video_path))
    skeleton_frames = []
    frame_indices = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            joints = results.pose_landmarks.landmark
            frame_data = [frame_idx]  # prepend frame index
            for joint in joints:
                frame_data.extend([joint.x, joint.y, joint.z])
            skeleton_frames.append(frame_data)
            frame_indices.append(frame_idx)

        frame_idx += 1

    cap.release()
    pose.close()

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for frame in skeleton_frames:
            writer.writerow(frame)

    print(f"✅ Skeleton saved to: {output_csv_path}")
    return output_csv_path
