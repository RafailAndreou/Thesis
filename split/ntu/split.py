import os
import shutil
import random
from collections import defaultdict
import re

# ==== CONFIG (change these for your dataset) ====
input_dir = r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\pseudo-rgb'
output_base_dir = r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\split_dataset2'

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Random seed for reproducibility
random.seed(42)

def extract_action_from_filename(filename):
    """
    Extract action number from filename like S004C002P003R001A030.skeleton.png
    Returns the action number (e.g., 30 from A030)
    """
    match = re.search(r'A(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def create_directory_structure(base_dir, actions):
    """
    Create train/val/test directory structure with action subdirectories
    """
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for action in actions:
            action_dir = os.path.join(split_dir, f'action_{action:03d}')
            os.makedirs(action_dir, exist_ok=True)

def split_dataset():
    """
    Split the dataset into train/val/test sets
    """
    print(f"Starting dataset split from {input_dir}...")
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    # Collect all images recursively
    all_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.png'):
                all_files.append(os.path.join(root, f))

    if not all_files:
        print("No PNG files found in input directory")
        return

    print(f"Found {len(all_files)} PNG files")

    # Group files by action class
    action_files = defaultdict(list)
    for file_path in all_files:
        filename = os.path.basename(file_path)
        action = extract_action_from_filename(filename)
        if action is not None:
            action_files[action].append(file_path)
        else:
            print(f"Warning: Could not extract action from filename: {filename}")

    print(f"\nFound {len(action_files)} different action classes")

    # Create directory structure
    create_directory_structure(output_base_dir, action_files.keys())

    total_train = total_val = total_test = 0

    for action, files in action_files.items():
        random.shuffle(files)
        n_files = len(files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        n_test = n_files - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]

        action_dir_name = f'action_{action:03d}'

        for file in train_files:
            shutil.copy2(file, os.path.join(output_base_dir, 'train', action_dir_name))
        for file in val_files:
            shutil.copy2(file, os.path.join(output_base_dir, 'val', action_dir_name))
        for file in test_files:
            shutil.copy2(file, os.path.join(output_base_dir, 'test', action_dir_name))

        print(f"Action {action:03d}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

    print(f"\nDataset split completed!")
    print(f"Total: {total_train + total_val + total_test}")
    print(f"Train: {total_train} ({total_train/(total_train+total_val+total_test)*100:.1f}%)")
    print(f"Val: {total_val} ({total_val/(total_train+total_val+total_test)*100:.1f}%)")
    print(f"Test: {total_test} ({total_test/(total_train+total_val+total_test)*100:.1f}%)")
    print(f"\nOutput directory: {output_base_dir}")

def create_class_mapping():
    """
    Create a mapping file for action numbers to class indices
    """
    train_dir = os.path.join(output_base_dir, 'train')
    action_dirs = sorted(d for d in os.listdir(train_dir) if d.startswith('action_'))
    mapping_file = os.path.join(output_base_dir, 'class_mapping.txt')

    with open(mapping_file, 'w') as f:
        f.write("Class Index -> Action Number\n" + "="*30 + "\n")
        for i, action_dir in enumerate(action_dirs):
            action_num = int(action_dir.split('_')[1])
            f.write(f"{i} -> {action_num}\n")

    print(f"Class mapping saved to: {mapping_file}")

if __name__ == "__main__":
    split_dataset()
    create_class_mapping()
