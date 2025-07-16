# full_pipeline.py
from video_to_csv import extract_skeleton_from_video
from visualize_csv import connections
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from tkinter import Tk, filedialog


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

def read_csv(csv_path):
    skeleton_frames = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            frame = np.array(row, dtype=float).reshape(-1, 3)
            skeleton_frames.append(frame)
    return skeleton_frames

def extract_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    video_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frames.append(rgb)
    cap.release()
    return video_frames

def visualize_side_by_side(video_frames, skeleton_frames):
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
        frames=min(len(video_frames), len(skeleton_frames)),
        interval=12,
        blit=True,
        repeat=True
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    Tk().withdraw()
    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.gif")]
    )

    if not video_path:
        print("❌ No file selected.")
    else:
        # 1. Convert video → CSV
        csv_path = extract_skeleton_from_video(video_path)

        # 2. Read both CSV and video frames
        skeleton_data = read_csv(csv_path)
        video_data = extract_video_frames(video_path)

        # 3. Show side-by-side
        visualize_side_by_side(video_data, skeleton_data)
