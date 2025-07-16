# visualize_csv.py

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# === MediaPipe pose connections ===

savepath = "C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\test.gif"
connections = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (12, 14), (14, 16), (11, 13), (13, 15),
    (12, 24), (24, 26), (26, 28), (28, 32), (32, 30),
    (11, 23), (23, 25), (25, 27), (27, 31), (31, 29),
    (16, 18), (18, 20), (20, 22), (15, 17), (17, 19), (19, 21), (12, 1),
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

# === Load skeleton + frame index from CSV ===
def load_skeleton_from_csv(csv_path):
    skeleton_frames = []
    frame_indices = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            frame_idx = int(row[0])
            joints = np.array(row[1:], dtype=float).reshape(-1, 3)
            skeleton_frames.append(joints)
            frame_indices.append(frame_idx)
    return skeleton_frames, frame_indices

# === Visualize side-by-side or just skeleton ===
def visualize_skeleton(skeleton_frames, frame_indices=None, video_frames=None):
    fig, axes = plt.subplots(1, 2 if video_frames else 1, figsize=(12, 6))
    if not video_frames:
        axes = [axes]

    ax2 = axes[-1]
    ax2.axis('off')
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_aspect('equal')

    if video_frames:
        ax1 = axes[0]
        ax1.axis('off')
        vid_image = ax1.imshow(np.zeros_like(video_frames[0]))
    else:
        vid_image = None

    lines = []
    for start, end in connections:
        color = get_color(start, end)
        line, = ax2.plot([], [], 'o-', lw=2, color=color)
        lines.append(line)

    def update(i):
        joints = skeleton_frames[i]

        if vid_image and frame_indices:
            video_idx = frame_indices[i]
            if video_idx < len(video_frames):
                vid_image.set_array(video_frames[video_idx])

        x = joints[:, 0] - np.mean(joints[:, 0])
        y = -joints[:, 1] + np.mean(joints[:, 1])

        for j, (start, end) in enumerate(connections):
            lines[j].set_data([x[start], x[end]], [y[start], y[end]])

        return lines + ([vid_image] if vid_image else [])

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(skeleton_frames),
        interval=12,
        blit=True,
        repeat=True
    )

    plt.tight_layout()
    if save_path:
        ani.save(save_path, writer='pillow', fps=30)
        print(f"✅ Animation saved to {save_path}")
    else:
        plt.show()

# === Standalone usage ===
if __name__ == "__main__":
    from tkinter import Tk, filedialog
    Tk().withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select a skeleton CSV file",
        filetypes=[("CSV files", "*.csv")]
    )

    if csv_path:
        skeleton_frames, frame_indices = load_skeleton_from_csv(csv_path)
        visualize_skeleton(skeleton_frames, frame_indices)
    else:
        print("❌ No file selected.")
