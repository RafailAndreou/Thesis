import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate

# Parameters
num = 159  # target number of frames
input_dirs = [
    r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\M',
    r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\L',
    r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\R'
]

output_dir = r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\fft_images2'
os.makedirs(output_dir, exist_ok=True)

# Read CSV
def readcsv(filename):
    with open(filename, newline='') as ifile:
        reader = csv.reader(ifile)
        return [row for row in reader]

# Process one CSV file
def process_csv(csv_filename):
    print('Processing:', csv_filename)
    try:
        picture = readcsv(csv_filename)
        signal_shape = len(picture)

        if signal_shape == 0:
            print(f"Empty file: {csv_filename}")
            return

        cols = len(picture[0])
        if cols != 150:
            print(f"Skipping {csv_filename} (expected 150 columns, got {cols})")
            return

        # Extract person 1 and person 2, then average them
        p1 = [list(map(float, row[0:75])) for row in picture]
        p2 = [list(map(float, row[75:150])) for row in picture]
        avg = np.mean([p1, p2], axis=0)
        signal_img = np.array(avg)  # shape: (frames, 75)

        # Transpose to shape (75, frames)
        c = signal_img.T

        # Interpolate to fixed number of frames (num)
        interpolated = []
        for k in range(75):
            arr = c[k]
            interp_func = interpolate.interp1d(np.arange(len(arr)), arr, kind='linear')
            stretched = interp_func(np.linspace(0, len(arr) - 1, num))
            interpolated.append(stretched)
        z = np.array(interpolated).T  # shape (num, 75)

        # FFT
        fft_array = np.fft.fft2(z)
        fft_shift = np.fft.fftshift(fft_array)
        magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1e-8)

        # Normalize to [0, 255]
        norm_mag = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum))
        image = (norm_mag * 255).astype(np.uint8)

        # Convert to RGB
        rgb_image = np.stack([image] * 3, axis=-1)  # shape (num, 75, 3)

        # Save
        base_filename = os.path.splitext(os.path.basename(csv_filename))[0]
        output_filename = os.path.join(output_dir, base_filename + '.png')
        matplotlib.pyplot.imsave(output_filename, rgb_image)
        print(f"Saved FFT image: {output_filename}")

    except Exception as e:
        print(f"Error processing {csv_filename}: {e}")

# Run on all directories
total_files = 0
processed_files = 0

for input_dir in input_dirs:
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        continue

    print(f"\nProcessing directory: {input_dir}")
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    total_files += len(csv_files)

    for csv_file in csv_files:
        csv_path = os.path.join(input_dir, csv_file)
        process_csv(csv_path)
        processed_files += 1

print(f"\nProcessing complete!")
print(f"Total files found: {total_files}")
print(f"Files processed: {processed_files}")
print(f"Output directory: {output_dir}")
