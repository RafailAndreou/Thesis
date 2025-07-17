import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# === CONFIG ===
input_path = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\R\S001C003P001R001A001.skeleton.csv"  # change this
output_dir = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\fft_image"
basename = os.path.splitext(os.path.basename(input_path))[0]
output_path = os.path.join(output_dir, f"{basename}.png")
interpolated_frame_count = 159

# === LOAD AND CLEAN ===
df = pd.read_csv(input_path)

# Drop frame index if present
if df.columns[0].lower().startswith("frame") or df.columns[0] == "0":
    df = df.iloc[:, 1:]

# Drop all-zero columns (unused joints)
df = df.loc[:, (df != 0).any(axis=0)]

data = df.to_numpy()
num_frames, num_columns = data.shape

# Ensure shape is divisible by 3
if num_columns % 3 != 0:
    raise ValueError(f"Expected joint data in x,y,z triplets. Got {num_columns} columns.")

num_joints = num_columns // 3

# Reshape to [frames, joints, 3]
data = data.reshape((num_frames, num_joints, 3))

# === INTERPOLATE ===
interp_data = np.zeros((interpolated_frame_count, num_joints, 3))
for j in range(num_joints):
    for d in range(3):
        series = data[:, j, d]
        interp_fn = interp1d(np.arange(num_frames), series, kind='linear', fill_value='extrapolate')
        interp_data[:, j, d] = interp_fn(np.linspace(0, num_frames - 1, interpolated_frame_count))

# === FLATTEN to 2D: shape (frames x coords) = [159 x (joints * 3)] ===
flat_data = interp_data.reshape(interpolated_frame_count, num_joints * 3)

# === 2D FFT ===
fft_2d = np.fft.fft2(flat_data)
fft_shifted = np.fft.fftshift(fft_2d)
magnitude = np.abs(fft_shifted)

# === LOG SCALE + Normalize to 0–255 ===
magnitude_log = 20 * np.log1p(magnitude)  # log1p avoids log(0)
magnitude_norm = (magnitude_log - magnitude_log.min()) / (magnitude_log.max() - magnitude_log.min() + 1e-8)
fft_img = (magnitude_norm * 255).astype(np.uint8)

# === SAVE IMAGE ===
plt.imsave(output_path, fft_img, cmap='gray')
print(f"✅ Saved FFT image to: {output_path}")
