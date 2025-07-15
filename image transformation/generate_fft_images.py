import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from PIL import Image
from pathlib import Path

# === CONFIGURATION ===
INPUT_ROOT = Path(
    r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET"
)
OUTPUT_ROOT = Path(
    r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\nku\fft_images"
)
IMAGE_SIZE = (224, 224)

# Final joints used (from cleaned connection list)
joints_to_use = sorted(set([
    0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20
]))

def read_skeleton_csv(file_path):
    frames = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 75:
                continue
            joints = np.array(row[:75], dtype=np.float32).reshape(-1, 3)
            frames.append(joints)
    return np.array(frames)

def compute_fft_image(skeleton_frames):
    selected = skeleton_frames[:, joints_to_use, :]  # shape: (T, J, 3)

    T, J, C = selected.shape
    fft_data = np.zeros((J, C, T), dtype=np.float32)

    for j in range(J):
        for c in range(C):
            signal = selected[:, j, c]
            fft_result = np.abs(fft(signal))
            fft_data[j, c, :] = fft_result

    # Normalize and stack as RGB
    rgb = np.stack([fft_data[:, 0, :], fft_data[:, 1, :], fft_data[:, 2, :]], axis=-1)  # (J, T, 3)
    rgb -= rgb.min()
    rgb /= (rgb.max() + 1e-6)
    rgb = (rgb * 255).astype(np.uint8)

    img = Image.fromarray(rgb).resize(IMAGE_SIZE)
    return img

def extract_action_label(filename):
    parts = filename.split('A')
    if len(parts) > 1:
        label_part = parts[1][:3]
        return f"A{label_part}"
    return "Unknown"

def process_all_files():
    for split in ["L", "M", "R"]:
        folder = INPUT_ROOT / split
        for file in folder.glob("*.skeleton.csv"):
            try:
                skeleton_frames = read_skeleton_csv(file)
                if skeleton_frames.shape[0] < 10:
                    continue

                img = compute_fft_image(skeleton_frames)
                label = extract_action_label(file.name)
                output_dir = OUTPUT_ROOT / label
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / (file.stem + ".png")
                img.save(output_path)
            except Exception as e:
                print(f"❌ Failed to process {file.name}: {e}")

if __name__ == "__main__":
    process_all_files()
    print("✅ All FFT images generated.")
