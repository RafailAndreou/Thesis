import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === CONFIGURATION ===
skeleton_csv_path = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\M\S017C002P020R002A022.skeleton.csv"
gif_output_path = "test.gif"

# NTU 25-joint default (uncleaned) connection list
connections = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 8), (8, 9), (9, 10),
    (20, 4), (4, 5), (5, 6),
    (20, 12), (12, 13), (13, 14),
    (0, 16), (16, 17), (0, 18), (18, 19)
]

def read_skeleton_csv(path):
    frames = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            joints = np.array(row[:75], dtype=np.float32).reshape(-1, 3)
            frames.append(joints)
    return frames

# Load data
frames = read_skeleton_csv(skeleton_csv_path)
frames = frames[:300]  # Optional limit

# === Plot Setup ===
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axis('off')
ax.set_aspect('equal')

lines = [ax.plot([], [], 'ro-', lw=2)[0] for _ in connections]

def update(frame_idx):
    joints = frames[frame_idx]
    x = joints[:, 0]
    y = joints[:, 1]
    for i, (start, end) in enumerate(connections):
        lines[i].set_data([x[start], x[end]], [y[start], y[end]])
    return lines

ani = animation.FuncAnimation(
    fig, update,
    frames=len(frames),
    interval=30,
    blit=True,
    repeat=True
)

# === Save as GIF ===
ani.save(gif_output_path, writer='pillow', fps=30)
print(f"✅ Animation saved as: {gif_output_path}")

plt.show()
