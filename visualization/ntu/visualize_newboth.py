import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === CONFIGURATION ===
skeleton_csv_path = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\L\S017C001P020R002A057.skeleton.csv"

# Final clean joint connections
connections = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (5, 6), (6, 7), (20, 4), (4, 5),
    (20, 8), (8, 9), (9, 10),
    (12, 13), (13, 14),
    (16, 17), (17, 18),
]

# Define coloring
color_map = {
    "spine": 'black',
    "left": 'blue',
    "right": 'red',
    "person2": 'green'
}

def get_connection_color(start, end, person_idx):
    if person_idx == 1:
        return color_map["person2"]
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
            data = np.array(row, dtype=np.float32)
            people_count = len(data) // 75
            joints_per_person = [data[i * 75:(i + 1) * 75].reshape(-1, 3) for i in range(people_count)]
            frames.append(joints_per_person)
    return frames

frames = read_skeleton_csv(skeleton_csv_path)

# === Plot Setup ===
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axis('off')
ax.set_aspect('equal')

# Create lines and labels per person
lines_all = []
labels_all = []

for person_idx in range(2):  # Support for up to 2 people
    lines = []
    for start, end in connections:
        color = get_connection_color(start, end, person_idx)
        line, = ax.plot([], [], 'o-', lw=2, color=color)
        lines.append(line)
    labels = [ax.text(0, 0, str(i), fontsize=8, color='gray') for i in range(25)]
    lines_all.append(lines)
    labels_all.append(labels)

# Animation update
def update(frame_idx):
    joints_list = frames[frame_idx]
    ax.set_title("2D Skeleton Animation")

    for person_idx, joints in enumerate(joints_list):
        x = joints[:, 0]
        y = joints[:, 1]
        lines = lines_all[person_idx]
        labels = labels_all[person_idx]

        for i, (start, end) in enumerate(connections):
            lines[i].set_data([x[start], x[end]], [y[start], y[end]])

        for i, label in enumerate(labels):
            label.set_position((x[i], y[i]))
    
    # Hide unused second person if only one is present
    for person_idx in range(len(joints_list), 2):
        for line in lines_all[person_idx]:
            line.set_data([], [])
        for label in labels_all[person_idx]:
            label.set_position((-10, -10))

    return sum(lines_all, []) + sum(labels_all, [])

ani = animation.FuncAnimation(
    fig, update,
    frames=len(frames),
    interval=14,
    blit=True,
    repeat=True
)

plt.show()
