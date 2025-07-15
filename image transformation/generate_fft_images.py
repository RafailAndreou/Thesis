import os
import csv
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from pathlib import Path

# === CONFIGURATION ===
INPUT_ROOT = Path(
    r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET"
)
OUTPUT_ROOT = Path(
    r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\nku\fft_images"
)
INTERPOLATED_LENGTH = 159  # Number of frames to interpolate to
SELECTED_JOINTS = sorted(set([
    0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20
]))

# === FUNCTIONS ===
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

def interpolate_to_fixed_length(frames, target_len=159):
    selected = frames[:, SELECTED_JOINTS, :]  # shape: (T, J, 3)
    T, J, C = selected.shape
    flattened = selected.reshape(T, -1).T  # shape: (J×3, T)

    interpolated = []
    for signal in flattened:
        interp_func = interpolate.interp1d(np.arange(T), signal, kind='linear', fill_value="extrapolate")
        stretched = interp_func(np.linspace(0, T - 1, target_len))
        interpolated.append(stretched)

    result = np.stack(interpolated, axis=1)  # shape: (target_len, J×3)
    return result

def fft_grayscale_image(matrix_2d):
    fft_array = np.fft.fft2(matrix_2d)
    fft_shift = np.fft.fftshift(fft_array)
    magnitude = 20 * np.log(np.abs(fft_shift) + 1e-8)
    normed = 255 * magnitude / np.max(magnitude)
    return normed.astype(np.uint8)

def extract_action_label(filename):
    parts = filename.split('A')
    if len(parts) > 1:
        return f"A{parts[1][:3]}"
    return "Unknown"

def process_all_files():
    for split in ["L", "M", "R"]:
        folder = INPUT_ROOT / split
        for file in folder.glob("*.skeleton.csv"):
            try:
                frames = read_skeleton_csv(file)
                if frames.shape[0] < 2:
                    continue
                interp = interpolate_to_fixed_length(frames, INTERPOLATED_LENGTH)
                fft_img = fft_grayscale_image(interp)

                label = extract_action_label(file.name)
                output_dir = OUTPUT_ROOT / label
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / (file.stem + ".png")
                plt.imsave(out_path, fft_img, cmap='gray')
            except Exception as e:
                print(f"❌ Failed to process {file.name}: {e}")

# === RUN ===
if __name__ == "__main__":
    process_all_files()
    print("✅ All grayscale FFT images generated.")
