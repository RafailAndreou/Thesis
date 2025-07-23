# ntu_skeleton_extractor.py

import cv2
import csv
import numpy as np
import mediapipe as mp
from pathlib import Path

JOINT_DIM = 3
NTU_PERSON_JOINTS = 25
SCALE_FACTOR = 4.0

def detect_num_people(segmentation_mask, threshold=0.25):
    if segmentation_mask is None:
        return 1
    binary_mask = segmentation_mask > 0.1
    ratio = np.sum(binary_mask) / binary_mask.size
    return 2 if ratio > threshold else 1

def extract_ntu_skeleton_from_video(video_path, output_path=None, verbose=True):
    """
    Extracts NTU-style skeleton data from a video using MediaPipe Holistic.

    Outputs:
    - If 1 person detected: 25 joints × 3 = 75 values per row
    - If 2 people detected: 50 joints × 3 = 150 values per row
    """
    video_path = Path(video_path)
    if output_path is None:
        output_csv_path = video_path.with_suffix(".ntu_skeleton.csv")
    else:
        output_csv_path = Path(output_path)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(str(video_path))

    skeleton_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        if results.pose_landmarks:
            joints = results.pose_landmarks.landmark[:NTU_PERSON_JOINTS]
            person1 = [
                coord * SCALE_FACTOR
                for joint in joints
                for coord in (joint.x, joint.y, joint.z)
            ]

            num_people = detect_num_people(getattr(results, "segmentation_mask", None))

            if num_people == 1:
                skeleton_row = person1
            else:
                # Placeholder: duplicate person1
                person2 = person1.copy()
                skeleton_row = person1 + person2

            skeleton_frames.append(skeleton_row)

    cap.release()
    holistic.close()

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(skeleton_frames)

    if verbose:
        print(f"✅ NTU-style skeleton saved to: {output_csv_path}")
    return output_csv_path

# === Standalone usage ===
if __name__ == "__main__":
    from tkinter import Tk, filedialog
    Tk().withdraw()
    selected_video = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.gif")]
    )
    if selected_video:
        extract_ntu_skeleton_from_video(selected_video)
    else:
        print("❌ No file selected.")
