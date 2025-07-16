import cv2
import csv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from tkinter import Tk, filedialog

# === FILE PICKER ===
Tk().withdraw()
video_path = filedialog.askopenfilename(
    title="Select a video file",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.gif")]
)
if not video_path:
    print("❌ No file selected.")
    exit()

video_path = Path(video_path)
output_csv_path = video_path.with_suffix(".skeleton.csv")

# === MediaPipe Pose Setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# === Read Video and Extract Skeletons ===
cap = cv2.VideoCapture(str(video_path))
skeleton_frames = []
video_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        joints = results.pose_landmarks.landmark
        frame_data = []
        for joint in joints:  # 33 joints
            frame_data.extend([joint.x, joint.y, joint.z])
        skeleton_frames.append(np.array(frame_data).reshape(-1, 3))
        video_frames.append(rgb)

cap.release()
pose.close()

# === Save CSV ===
with open(output_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    for frame in skeleton_frames:
        writer.writerow(frame.flatten())

print(f"✅ Skeleton saved to: {output_csv_path}")

# === MediaPipe Connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (12, 14), (14, 16), (11, 13), (13, 15),
    (12, 24), (24, 26), (26, 28), (28, 32), (32, 30),
    (11, 23), (23, 25), (25, 27), (27, 31), (31, 29),
    (16, 18), (18, 20), (20, 22), (15, 17), (17, 19), (19, 21),(12,1),
]

def get_color(start, end):
    left = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
    right = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
    face = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    if start in left or end in left:
        return 'blue' if start < 24 else 'green'
    elif start in right or end in right:
        return 'red' if start < 24 else 'orange'
    elif start in face or end in face:
        return 'gray'
    else:
        return 'black'

# === Plot Setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.axis('off')
ax2.axis('off')
ax2.set_xlim(-0.6, 0.6)
ax2.set_ylim(-0.6, 0.6)
ax2.set_aspect('equal')

vid_image = ax1.imshow(np.zeros_like(video_frames[0]))
lines = []
for start, end in connections:
    color = get_color(start, end)
    line, = ax2.plot([], [], 'o-', lw=2, color=color)
    lines.append(line)

# === Animation Update
def update(frame_idx):
    frame = video_frames[frame_idx]
    joints = skeleton_frames[frame_idx]

    x = joints[:, 0] - np.mean(joints[:, 0])
    y = -joints[:, 1] + np.mean(joints[:, 1])

    vid_image.set_array(frame)
    for i, (start, end) in enumerate(connections):
        lines[i].set_data([x[start], x[end]], [y[start], y[end]])
    return lines + [vid_image]

ani = animation.FuncAnimation(
    fig, update,
    frames=len(skeleton_frames),
    interval=33,
    blit=True,
    repeat=True
)

plt.tight_layout()
plt.show()
