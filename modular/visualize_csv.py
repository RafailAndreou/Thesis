# visualize_csv.py
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tkinter import Tk, filedialog

# Define connections (same as MediaPipe)
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

def visualize_csv(csv_path):
    skeleton_frames = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            frame = np.array(row, dtype=float).reshape(-1, 3)
            skeleton_frames.append(frame)

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal')

    lines = []
    for start, end in connections:
        color = get_color(start, end)
        line, = ax.plot([], [], 'o-', lw=2, color=color)
        lines.append(line)

    def update(frame_idx):
        joints = skeleton_frames[frame_idx]
        x = joints[:, 0] - np.mean(joints[:, 0])
        y = -joints[:, 1] + np.mean(joints[:, 1])
        for i, (start, end) in enumerate(connections):
            lines[i].set_data([x[start], x[end]], [y[start], y[end]])
        return lines

    ani = animation.FuncAnimation(
        fig, update, frames=len(skeleton_frames),
        interval=33, blit=True, repeat=True
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    Tk().withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select a skeleton CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    if csv_path:
        visualize_csv(csv_path)
    else:
        print("❌ No file selected.")
