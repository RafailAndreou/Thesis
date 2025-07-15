import cv2
import csv
import numpy as np
import mediapipe as mp
from pathlib import Path
from tkinter import Tk, filedialog

# === CONFIGURATION ===
NUM_NTU_JOINTS = 25
OUTPUT_SUFFIX = ".skeleton.csv"

# === NTU to MediaPipe Joint Mapping (approximate) ===
ntu_to_mp = {
    0: 24,   # Hip center
    1: 23,   # Spine base
    2: 11,   # Mid spine
    3: 12,   # Upper spine
    4: 11,   # Left shoulder root (approx)
    5: 13,   # Left shoulder
    6: 15,   # Left elbow
    7: 17,   # Left wrist
    8: 14,   # Right shoulder
    9: 16,   # Right elbow
    10:18,   # Right wrist
    12:25,   # Left hip
    13:27,   # Left knee
    14:31,   # Left ankle
    16:26,   # Right hip
    17:28,   # Right knee
    18:32,   # Right ankle
    20: 0    # Head top / neck base
}

# === FILE PICKER ===
Tk().withdraw()
file_path = filedialog.askopenfilename(
    title="Select a video file",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.gif")]
)
if not file_path:
    print("❌ No file selected.")
    exit()

VIDEO_PATH = Path(file_path)
OUTPUT_CSV = VIDEO_PATH.with_suffix(OUTPUT_SUFFIX)

# === MEDIAPIPE SETUP ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# === PROCESS VIDEO ===
cap = cv2.VideoCapture(str(VIDEO_PATH))
skeleton_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        frame_joints = []

        for j in range(NUM_NTU_JOINTS):
            if j in ntu_to_mp:
                mp_idx = ntu_to_mp[j]
                joint = landmarks[mp_idx]
                frame_joints.extend([joint.x, joint.y, joint.z])
            else:
                frame_joints.extend([0.0, 0.0, 0.0])  # Fill missing joints with zeros

        skeleton_frames.append(frame_joints)

cap.release()
pose.close()

# === SAVE TO CSV ===
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(skeleton_frames)

print(f"✅ Saved NTU-style skeleton to: {OUTPUT_CSV}")

