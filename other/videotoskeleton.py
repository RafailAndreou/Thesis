import cv2
import csv
import mediapipe as mp
from pathlib import Path
from tkinter import Tk, filedialog

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
OUTPUT_CSV = VIDEO_PATH.with_suffix(".skeleton.csv")

# === MediaPipe Pose Setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# === Read Video and Extract Skeletons ===
cap = cv2.VideoCapture(str(VIDEO_PATH))
skeleton_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        joints = results.pose_landmarks.landmark
        frame_data = []
        for joint in joints:  # 33 joints
            frame_data.extend([joint.x, joint.y, joint.z])
        skeleton_frames.append(frame_data)

cap.release()
pose.close()

# === Save to CSV ===
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(skeleton_frames)

print(f"✅ Saved MediaPipe skeleton to: {OUTPUT_CSV}")
