import os
import numpy as np
import csv
from scipy import interpolate
import matplotlib.pyplot as plt

# === CONFIG ===
input_dirs = [
    r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\pku\SkeletonData\actionsL",
    r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\pku\SkeletonData\actionsM",
    r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\pku\SkeletonData\actionsR"
]
output_dir = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\fft_pku_images"
num_frames = 159  # Interpolated length
num_joints = 25
dims_per_joint = 3  # x, y, z

os.makedirs(output_dir, exist_ok=True)

def read_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = [list(map(float, row[:75])) for row in reader if row]  # Only first 75 values (25 joints × 3)
    return np.array(data)

def interpolate_to_fixed_length(data, num_frames):
    interpolated = []
    data = data.T  # shape: 75 × T
    for joint_series in data:
        interp_func = interpolate.interp1d(np.arange(joint_series.size), joint_series, kind='linear', fill_value='extrapolate')
        stretched = interp_func(np.linspace(0, joint_series.size - 1, num_frames))
        interpolated.append(stretched)
    return np.array(interpolated).T  # shape: 159 × 75

def save_fft_image(interpolated_data, output_path):
    fft_array = np.fft.fft2(interpolated_data)
    fft_shift = np.fft.fftshift(fft_array)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1e-8)
    magnitude_norm = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.ptp())
    plt.imsave(output_path, magnitude_norm, cmap='gray')

# === PROCESS ALL FILES ===
for input_dir in input_dirs:
    for fname in os.listdir(input_dir):
        if not fname.endswith(".csv"):
            continue

        csv_path = os.path.join(input_dir, fname)
        try:
            raw_data = read_csv(csv_path)
            if raw_data.shape[0] < 5:
                print(f"⚠️ Skipping too-short file: {fname}")
                continue

            interpolated = interpolate_to_fixed_length(raw_data, num_frames)
            output_path = os.path.join(output_dir, fname.replace(".csv", ".png"))
            save_fft_image(interpolated, output_path)
            print(f"✅ Saved {os.path.basename(output_path)}")

        except Exception as e:
            print(f"❌ Failed on {fname}: {e}")
