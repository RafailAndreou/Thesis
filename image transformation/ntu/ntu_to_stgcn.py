import os
import numpy as np
import pandas as pd
import pickle
import re

# === CONFIG ===
input_root = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\nozero"
output_root = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\STGCN_data"
T = 300  # fixed sequence length (same as NTU ST-GCN standard)
V = 25   # joints
C = 3    # x,y,z
M = 1    # number of persons in your dataset

os.makedirs(output_root, exist_ok=True)

def extract_action(filename):
    """
    Extract action index (zero-based) from filename.
    Example: S001C001P001R001A001.skeleton.csv -> action 0
    """
    match = re.search(r'A(\d+)', filename)
    if match:
        return int(match.group(1)) - 1  # zero-based index
    else:
        raise ValueError(f"Action not found in filename: {filename}")

def load_skeleton_csv(path):
    """
    Load skeleton CSV and return numpy array (T, V, C)
    """
    df = pd.read_csv(path, header=None)
    data = df.to_numpy()
    # Keep only 25 joints × 3 coords
    if data.shape[1] >= V * C:
        data = data[:, :V*C]
    else:
        raise ValueError(f"Unexpected number of columns in {path}")
    frames = data.reshape((len(data), V, C))
    return frames

all_samples, labels = [], []

for view in ['L', 'M', 'R']:
    view_dir = os.path.join(input_root, view)
    for file in os.listdir(view_dir):
        if file.endswith('.csv'):
            action = extract_action(file)
            seq = load_skeleton_csv(os.path.join(view_dir, file))
            # Pad or trim to fixed length T
            if seq.shape[0] < T:
                pad = np.zeros((T - seq.shape[0], V, C))
                seq = np.concatenate([seq, pad], axis=0)
            else:
                seq = seq[:T]
            all_samples.append(seq)
            labels.append(action)

N = len(all_samples)
print(f"Total samples: {N}")

# Final tensor (N, C, T, V, M)
data = np.zeros((N, C, T, V, M), dtype=np.float32)
for i, seq in enumerate(all_samples):
    data[i, :, :, :, 0] = seq.transpose(2, 0, 1)

# Save dataset
np.save(os.path.join(output_root, 'train_data.npy'), data)
label_dict = {'sample_name': [f'sample_{i}' for i in range(N)], 'label': labels}
with open(os.path.join(output_root, 'train_label.pkl'), 'wb') as f:
    pickle.dump(label_dict, f)

print(f"Data shape: {data.shape}")
print(f"Labels saved: {len(labels)}")
print(f"Saved to: {output_root}")
