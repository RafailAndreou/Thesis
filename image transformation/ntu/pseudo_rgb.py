import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# === CONFIG ===
input_file = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\L\S001C001P001R001A001.skeleton.csv"  # <<< CHANGE THIS
output_dir = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\pseudo-rgb"
basename = os.path.splitext(os.path.basename(input_file))[0]
output_file = os.path.join(output_dir, f"{basename}.png")

# === LOAD CSV ===
df = pd.read_csv(input_file)

# Remove frame index if present
if df.columns[0].lower().startswith("frame") or df.columns[0] == "0":
    df = df.iloc[:, 1:]

# Remove fully-zero columns
df = df.loc[:, (df != 0).any(axis=0)]

data = df.to_numpy()
num_frames, num_cols = data.shape

if num_cols % 3 != 0:
    raise ValueError(f"Expected x,y,z triplets. Got {num_cols} columns.")

num_joints = num_cols // 3

# === RESHAPE: [frames, joints, 3] ===
data = data.reshape((num_frames, num_joints, 3))

# === NORMALIZE EACH CHANNEL SEPARATELY ===
data_min = data.min(axis=(0, 1), keepdims=True)  # shape: [1,1,3]
data_max = data.max(axis=(0, 1), keepdims=True)
data_norm = (data - data_min) / (data_max - data_min + 1e-8)

# === TRANSPOSE: [joints, frames, RGB] → image shape ===
pseudo_rgb = np.transpose(data_norm, (1, 0, 2))  # [H, W, 3]

# === SAVE IMAGE ===
plt.imsave(output_file, pseudo_rgb)
print(f"✅ Saved pseudo-RGB image to: {output_file}")
