import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# === CONFIGURATION ===
skeleton_csv_path = r"C:\Users\rafai\Desktop\videoplayback.skeleton.csv"  # <- Set your file path
gif_output_path = Path(skeleton_csv_path).with_suffix(".gif")

# === JOINT CONNECTIONS (based on your NTU mapping) ===
connections = [
    # Spine
    (0, 1), (1, 20), (20, 2), (2, 3),(0,2),(2,20),(1,3),(3,20),(0,20),(20,23),

    # Left arm
    (5, 6), (6, 7), (20, 4), (4, 5),

    # Right arm
    (20, 8), (8, 9), (9, 10),

    # Left leg
    (12, 13), (13, 14),(12,1),

    # Right leg
    (16, 17), (17, 18),(16,0),
]

# === COLORING ===
color_map = {
    "spine": 'black',
    "left": 'blue',
    "right": 'red'
}

def get_connection_color(start, end):
    left = {4, 5, 6, 7, 12, 13, 14}
    right = {8, 9, 10, 16, 17, 18}
    if start in left or end in left:
        return color_map["left"]
    elif start in right or end in right:
        return color_map["right"]
    else:
        return color_map["spine"]

# === LOAD CSV ===
def read_skeleton_csv(path):
    frames = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 75:
                joints = np.array(row, dtype=np.float32).reshape(-1, 3)
                frames.append(joints)
    return frames

frames = read_skeleton_csv(skeleton_csv_path)
frames = frames[:300]  # Optional trim

# === PLOT ===
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.6, 0.6)

ax.axis('off')
ax.set_aspect('equal')

lines = []
for start, end in connections:
    color = get_connection_color(start, end)
    line, = ax.plot([], [], 'o-', lw=2, color=color)
    lines.append(line)

labels = [ax.text(0, 0, str(i), fontsize=8, color='gray') for i in range(25)]

def update(frame_idx):
    joints = frames[frame_idx]
    x = joints[:, 0]
    y = -joints[:, 1]

    # Center the skeleton
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

