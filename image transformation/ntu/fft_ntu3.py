import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Parameters
num = 159
target_cols = 150  # always output 150 features per frame (75 or 150 padded)

# Input directories
input_dirs = [
    r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\M',
    r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\L',
    r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\R'
]

# Output directory
output_dir = r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\fft_images_rgb_adaptive'
os.makedirs(output_dir, exist_ok=True)

# Read CSV with adaptive column handling (1 or 2 people)
def read_csv_adaptive(filename, output_cols=150):
    data = []
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            row = [val.strip() for val in row if val.strip() != '']
            try:
                row_vals = [float(val) for val in row]
                if len(row_vals) >= 75:
                    # If < 150, pad with zeros to reach 150
                    padded = row_vals[:output_cols] + [0.0] * max(0, output_cols - len(row_vals))
                    data.append(padded)
            except ValueError:
                continue  # skip corrupted rows
    return np.array(data)

def process_csv(csv_filename):
    print('Processing:', csv_filename)
    try:
        signal_img = read_csv_adaptive(csv_filename, output_cols=target_cols)

        if signal_img.shape[0] < 5:
            print(f"Skipping {csv_filename}: too few valid frames.")
            return

        # Transpose to shape (joints, time)
        c = signal_img.T

        # Interpolate/stretch to fixed length
        interpolated = []
        for joint_series in c:
            f_interp = interpolate.interp1d(
                np.arange(len(joint_series)),
                joint_series,
                kind='linear',
                fill_value="extrapolate"
            )
            interpolated.append(f_interp(np.linspace(0, len(joint_series) - 1, num)))

        znew = np.array(interpolated).T  # Shape: (num, 150)

        # FFT
        fft_array = np.fft.fft2(znew)
        fft_shift = np.fft.fftshift(fft_array)
        magnitude = 20 * np.log(np.abs(fft_shift) + 1e-8)
        img = magnitude * 255 / np.max(magnitude)
        img = np.uint8(img)

        # Convert grayscale to RGB
        rgb_img = np.stack([img]*3, axis=-1)

        # Save image
        base_name = os.path.splitext(os.path.basename(csv_filename))[0]
        output_path = os.path.join(output_dir, base_name + '.png')
        plt.imsave(output_path, rgb_img)
        print(f'Saved RGB: {output_path}')

    except Exception as e:
        print(f'❌ Error with {csv_filename}: {e}')

# Batch process all CSVs
total_files = 0
processed_files = 0

for input_dir in input_dirs:
    print(f'\n📂 Processing directory: {input_dir}')
    if not os.path.exists(input_dir):
        print(f'⚠️ Skipped (not found): {input_dir}')
        continue

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    total_files += len(csv_files)
    for file in csv_files:
        full_path = os.path.join(input_dir, file)
        process_csv(full_path)
        processed_files += 1

print("\n✅ All done.")
print(f"Total files: {total_files}")
print(f"Processed successfully: {processed_files}")
print(f"Output dir: {output_dir}")
