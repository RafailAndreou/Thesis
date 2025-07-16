import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# === JOINT CONNECTIONS ===
connections = [
    # Spine
    (0, 1), (1, 20), (20, 2), (2, 3), (0, 2), (2, 20), (1, 3), (3, 20), (0, 20), (20, 23),

    # Left arm
    (5, 6), (6, 7), (20, 4), (4, 5),

    # Right arm
    (20, 8), (8, 9), (9, 10),

    # Left leg
    (12, 13), (13, 14), (12, 1),

    # Right leg
    (16, 17), (17, 18), (16, 0),
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

# === Load CSV ===
def read_skeleton_csv(path):
    frames = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 75:
                joints = np.array(row, dtype=np.float32).reshape(-1, 3)
                frames.append(joints)
    return frames

# === Main Visualization Function ===
def visualize_skeleton_from_csv(csv_path: Path, save_gif: bool = True):
    frames = read_skeleton_csv(csv_path)
    frames = frames[:300]

    gif_output_path = csv_path.with_suffix('.gif')

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

    if save_gif:
        ani.save(gif_output_path, writer='pillow', fps=30)
        print(f"✅ GIF saved as '{gif_output_path}'")

    plt.show()

# === RUN STANDALONE ===
if __name__ == "__main__":
    visualize_skeleton_from_csv(Path(r"C:\Users\rafai\Desktop\videoplayback.skeleton.csv"))
