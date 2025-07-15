import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# === CONFIGURATION ===
skeleton_csv_path = r"C:\Users\rafai\Desktop\videoplayback.skeleton.csv"  # Set your path
gif_output_path = Path(skeleton_csv_path).with_suffix(".gif")

# === MediaPipe Pose Connections (33 joints) ===
# Format: (start_joint, end_joint)
connections = [
    (0, 1), (1, 2), (2, 3), (3, 7),       # Nose → head
    (0, 4), (4, 5), (5, 6), (6, 8),       # Right eye, ear
    (0, 9), (9, 10), (10, 11), (11, 12),  # Left eye, ear
    (12, 14), (14, 16),                  # Left arm
    (11, 13), (13, 15),                  # Right arm
    (12, 24), (11, 23),                  # Shoulders to hips
    (24, 26), (26, 28),                  # Left leg
    (23, 25), (25, 27),                  # Right leg
    (28, 32), (27, 31),                  # Ankles to feet
    (32, 30), (31, 29),                  # Toes
    (16, 18), (15, 17),                  # Wrists to hands
    (18, 20), (17, 19),                  # Hands to tips
    (20, 22), (19, 21)                   # Tips to pinky/thumb
]

# === Load CSV ===
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

# === Plot Setup ===
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.6, 0.6)
ax.axis('off')
ax.set_aspect('equal')

lines = []
for (start, end) in connections:
    line, = ax.plot([], [], 'o-', lw=2, color='blue')
    lines.append(line)

labels = [ax.text(0, 0, str(i), fontsize=6, color='gray') for i in range(33)]

# === Animation ===
def update(frame_idx):
    joints = frames[frame_idx]
    x = joints[:, 0]
    y = -joints[:, 1]  # Flip Y axis

    # Centering
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
