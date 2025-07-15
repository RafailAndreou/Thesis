
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === CONFIGURATION ===
skeleton_csv_path = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\M\S017C002P020R002A022.skeleton.csv"

# Final clean joint connections
connections = [
    # Spine
    (0, 1), (1, 20), (20, 2), (2, 3),

    # Left arm
    (20, 5), (5, 6), (6, 7),

    # Right arm
    (20, 8), (8, 9), (9, 10),

    # Left leg
    (12, 13), (13, 14),

    # Right leg
    (16, 17), (17, 18),
]

# Define coloring
color_map = {
    "spine": 'black',
    "left": 'blue',
    "right": 'red'
}

# Assign color to each connection
def get_connection_color(start, end):
    left = {5, 6, 7, 11, 12, 13, 14}
    right = {8, 9, 10, 15, 16, 17, 18}
    if start in left or end in left:
        return color_map["left"]
    elif start in right or end in right:
        return color_map["right"]
    else:
        return color_map["spine"]

# Load CSV
def read_skeleton_csv(path):
    frames = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            joints = np.array(row[:75], dtype=np.float32).reshape(-1, 3)
            frames.append(joints)
    return frames

frames = read_skeleton_csv(skeleton_csv_path)
frames = frames[:300]

# === Plot Setup ===
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axis('off')
ax.set_aspect('equal')

# Create colored lines for each connection
lines = []
for start, end in connections:
    color = get_connection_color(start, end)
    line, = ax.plot([], [], 'o-', lw=2, color=color)
    lines.append(line)

# Create joint index labels
labels = [ax.text(0, 0, str(i), fontsize=8, color='gray') for i in range(25)]

# Animation update
def update(frame_idx):
    joints = frames[frame_idx]
    x = joints[:, 0]
    y = joints[:, 1]

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

plt.show()
