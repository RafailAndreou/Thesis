import os
import numpy as np
import pandas as pd
import pickle
import re

# === CONFIG ===
input_root = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\nozero"
output_root = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\STGCN_data"
T = 300  # fixed sequence length
V = 25   # joints
C = 3    # x,y,z
M = 1    # number of persons (1 for now)

os.makedirs(output_root, exist_ok=True)

def extract_action(filename):
    """
    Extract action index (zero-based) from filename.
    Example: S001C001P001R001A001.skeleton.csv -> action 0
    """
    match = re.search(r'A(\d+)', filename)
    if match:
        return int(match.group(1)) - 1  # zero-based
    else:
        raise ValueError(f"Action not found in filename: {filename}")

def load_skeleton_csv(path):
    """
    Load skeleton CSV and return numpy array (T, V, C)
    """
    df = pd.read_csv(path, header=None)
    data = df.to_numpy()
    # Keep only first 25 joints (75 values)
    if data.shape[1] >= V * C:
        data = data[:, :V * C]
    else:
        raise ValueError(f"Unexpected number of columns in {path}")
    frames = data.reshape((len(data), V, C))
    return frames

all_samples, labels = [], []

print("Scanning input folders...")
for view in ['L', 'M', 'R']:
    view_dir = os.path.join(input_root, view)
    if not os.path.exists(view_dir):
        print(f"❌ View folder not found: {view_dir}")
        continue

    for file in os.listdir(view_dir):
        if file.endswith('.csv'):
            action = extract_action(file)
            seq = load_skeleton_csv(os.path.join(view_dir, file))

            # Pad or cut to fixed length
            if seq.shape[0] < T:
                pad = np.zeros((T - seq.shape[0], V, C))
                seq = np.concatenate([seq, pad], axis=0)
            else:
                seq = seq[:T]

            all_samples.append(seq)
            labels.append(action)
            print(f"Processed: {file} -> action {action}")

N = len(all_samples)
if N == 0:
    print("❌ No samples found! Check input_root path and CSV files.")
    exit()

# Convert to ST-GCN format: (N, C, T, V, M)
print(f"\nBuilding dataset with {N} samples...")
data = np.zeros((N, C, T, V, M), dtype=np.float32)
for i, seq in enumerate(all_samples):
    data[i, :, :, :, 0] = seq.transpose(2, 0, 1)

# === Save dataset ===
print("\nSaving dataset...")
with open(os.path.join(output_root, 'train_data.npy'), 'wb') as f:
    np.save(f, data)

label_dict = {'sample_name': [f'sample_{i}' for i in range(N)], 'label': labels}
with open(os.path.join(output_root, 'train_label.pkl'), 'wb') as f:
    pickle.dump(label_dict, f)

print(f"\n✅ Saved {N} samples")
print(f"Data shape: {data.shape}")
print(f"Labels saved: {len(labels)}")
print(f"Output folder: {output_root}")
