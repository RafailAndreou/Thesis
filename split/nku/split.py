import os
import shutil
import random
from collections import defaultdict

# Input and output
input_root = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\fft_image"
output_root = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\split_dataset"

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Set seed for reproducibility
random.seed(42)

# Prepare action-based grouping
action_files = defaultdict(list)

# Scan L, M, R folders
for view in ['L', 'M', 'R']:
    view_dir = os.path.join(input_root, view)
    for filename in os.listdir(view_dir):
        if filename.endswith(".png"):
            # Extract action number (e.g. A032 → 32)
            parts = filename.split("A")
            if len(parts) < 2:
                continue
            action_str = parts[1][:3]
            if not action_str.isdigit():
                continue
            action_id = int(action_str)
            if 1 <= action_id <= 60:
                filepath = os.path.join(view_dir, filename)
                action_files[action_id].append(filepath)

# Create output folders
for split in ['train', 'val', 'test']:
    for cls in range(1, 61):
        os.makedirs(os.path.join(output_root, split, f"{cls:02}"), exist_ok=True)

# Split and copy
for action_id, files in action_files.items():
    random.shuffle(files)
    total = len(files)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]

    for f in train_files:
        shutil.copy(f, os.path.join(output_root, 'train', f"{action_id:02}", os.path.basename(f)))
    for f in val_files:
        shutil.copy(f, os.path.join(output_root, 'val', f"{action_id:02}", os.path.basename(f)))
    for f in test_files:
        shutil.copy(f, os.path.join(output_root, 'test', f"{action_id:02}", os.path.basename(f)))

    print(f"[✓] Action {action_id:02}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

print("\n✅ Dataset split complete.")
