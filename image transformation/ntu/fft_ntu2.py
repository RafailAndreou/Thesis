import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# === CONFIGURATION ===
input_file = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\M\S001C002P001R001A001.skeleton.csv"  # <<< CHANGE THIS
output_interpolated_dir = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\interprolated"
output_fft_dir = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\fft_image"
num = 159  # target frame length

# === Ensure output folders exist ===
os.makedirs(output_interpolated_dir, exist_ok=True)
os.makedirs(output_fft_dir, exist_ok=True)

# === READ CSV ===
def readcsv(filename):	
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return [row for row in reader]

# === Start processing ===
print(f"Processing: {input_file}")
filename = os.path.basename(input_file)
basename = os.path.splitext(filename)[0]

picture = readcsv(input_file)
signal_shape = len(picture)

# Convert to NumPy, take only first 75 columns
signal_img = np.zeros((signal_shape, 75))
for j in range(signal_shape):
    signal_img[j] = picture[j][0:75]

# Transpose to [75, frames] for interpolation
c = signal_img.T

# === Interpolation ===
b = []
for k in range(75):
    arr2 = np.array(c[k], dtype=np.float32)
    interp_fn = interpolate.interp1d(np.arange(arr2.size), arr2, kind='linear', fill_value='extrapolate')
    arr2_stretch = interp_fn(np.linspace(0, arr2.size - 1, num))
    b.extend(arr2_stretch)
z = np.reshape(b, (75, num)).T  # Shape: [159, 75]

# === Save interpolated image ===
interpolated_output_path = os.path.join(output_interpolated_dir, f"{basename}.png")
plt.imsave(interpolated_output_path, z, cmap='gray')
print(f"✅ Saved interpolated image to: {interpolated_output_path}")

# === FFT ===
fft_array = np.fft.fft2(z)
fft_shift = np.fft.fftshift(fft_array)
magnitude_spectrum = 20 * np.log1p(np.abs(fft_shift))  # log1p avoids log(0)

# Normalize to 0–255 and convert to uint8
magnitude_scaled = (magnitude_spectrum / magnitude_spectrum.max()) * 255
magnitude_scaled = magnitude_scaled.astype(np.uint8)

# === Save FFT image ===
fft_output_path = os.path.join(output_fft_dir, f"{basename}.png")
plt.imsave(fft_output_path, magnitude_scaled, cmap='gray')
print(f"✅ Saved FFT image to: {fft_output_path}")
