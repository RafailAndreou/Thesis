import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import fft
from sklearn.preprocessing import MinMaxScaler

# Input and output base directories
input_root = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\nozero"
output_root = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\fft_image"
os.makedirs(output_root, exist_ok=True)

# Resize shape for ResNet50
target_shape = (224, 224)

def normalize_and_convert_to_image(fft_matrix, shape):
    # 🔧 Amplify and log-scale to enhance contrast
    fft_magnitude = np.log1p(np.abs(fft_matrix) * 100)

    # Normalize each column (joint) to 0–255
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(fft_magnitude)
    norm_data = (norm_data * 255).astype(np.uint8)

    # Resize and convert to 3-channel RGB image
    img = Image.fromarray(norm_data)
    img = img.resize(shape, Image.BILINEAR)
    return img.convert("RGB")

# Traverse subfolders: L, M, R
for subfolder in ['L', 'M', 'R']:
    input_subdir = os.path.join(input_root, subfolder)
    output_subdir = os.path.join(output_root, subfolder)
    os.makedirs(output_subdir, exist_ok=True)

    for filename in os.listdir(input_subdir):
        if not filename.endswith(".csv"):
            continue

        input_path = os.path.join(input_subdir, filename)
        output_path = os.path.join(output_subdir, os.path.splitext(filename)[0] + ".png")

        try:
            df = pd.read_csv(input_path, header=None)
            data = df.to_numpy()

            # FFT across time (rows) for each joint (column)
            fft_result = fft(data, axis=0)
            fft_result = fft_result[:len(fft_result)//2]  # Use half-spectrum

            # Convert FFT to enhanced image
            img = normalize_and_convert_to_image(fft_result, target_shape)
            img.save(output_path)

            print(f"[✓] Saved image for: {subfolder}/{filename}")
        except Exception as e:
            print(f"[!] Error processing {subfolder}/{filename}: {e}")
