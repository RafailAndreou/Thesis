import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# === CONFIGURATION ===
skeleton_csv_path = r"C:\Users\rafai\Desktop\videoplayback.skeleton.csv"
gif_output_path = Path(skeleton_csv_path).with_suffix(".gif")

# === MediaPipe Connections (33 joints)
connections = [
    # Face/Head
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (0, 9), (9,10), (10,11), (11,12),

    # Arms
    (12,14), (14,16),     # Left
    (11,13), (13,15),     # Right
    (16,18), (18,20), (20,22),  # Left hand
    (15,17), (17,19), (19,21),(12,1),  # Right hand

    # Legs
    (12,24), (24,26), (26,28), (28,32), (32,30),  # Left leg
    (11,23), (23,25), (25,27), (27,31), (31,29),  # Right leg
]

# === Assign Color to Body Regions
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

# === Load CSV
def read_skeleton_csv(path):
    frames = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 99:
                joints = np.array(row, dtype=np.float32).reshape(-1, 3)
                frames.append(joints)
    return frames

frames = read_skeleton_csv(skeleton_csv_path)
frames = frames[:300]

# === Plot Setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.6, 0.6)
ax.axis('off')
ax.set_aspect('equal')

# Colored lines per connection
lines = []
for start, end in connections:
    color = get_color(start, end)
    line, = ax.plot([], [], 'o-', lw=2, color=color)
    lines.append(line)

# Joint labels
labels = [ax.text(0, 0, str(i), fontsize=6, color='gray') for i in range(33)]

# === Animation Function
def update(frame_idx):
    joints = frames[frame_idx]
    x = joints[:, 0]
    y = -joints[:, 1]  # Flip Y

    # Center
    x -= np.mean(x)
    y -= np.mean(y)

    for i, (start, end) in enumerate(connections):
        lines[i].set_data([x[start], x[end]], [y[start], y[end]])

    for i, label in enumerate(labels):
        label.set_position((x[i], y[i]))

    return lines + labels

ani = animation.FuncAnimation(
    fig, update,
    frames=len(frames),
    interval=14,
    blit=True,
    repeat=True
)

ani.save(gif_output_path, writer='pillow', fps=30)
print(f"✅ GIF saved as '{gif_output_path}'")

plt.show()
