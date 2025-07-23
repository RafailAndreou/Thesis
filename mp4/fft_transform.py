# fft_transform.py

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.fft import fft
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import os

def normalize_and_convert_to_image(fft_matrix, shape=(224, 224)):
    """
    Normalize and convert the FFT matrix to a pseudo-RGB image.
    """
    fft_magnitude = np.log1p(np.abs(fft_matrix) * 100)  # Enhance contrast
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(fft_magnitude)
    norm_data = (norm_data * 255).astype(np.uint8)

    img = Image.fromarray(norm_data)
    img = img.resize(shape, Image.BILINEAR)
    return img.convert("RGB")


def csv_to_fft_image(csv_path, output_path=None, image_size=(224, 224), verbose=True):
    """
    Converts a skeleton CSV file to an FFT image using the same logic as fft_ntu3.py.

    Parameters:
    - csv_path (str or Path): path to the input CSV
    - output_path (str or Path, optional): path to save the output image. If None, returns the image.
    - image_size (tuple): size of the final image (width, height)
    - verbose (bool): whether to print processing messages

    Returns:
    - PIL.Image object if output_path is None; otherwise, returns output_path
    """
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.with_suffix(".fft.png")
    else:
        output_path = Path(output_path)

    try:
        df = pd.read_csv(csv_path, header=None)
        data = df.iloc[:, 1:].to_numpy()  # skip frame index

        fft_result = fft(data, axis=0)
        fft_result = fft_result[:len(fft_result) // 2]  # keep half-spectrum

        img = normalize_and_convert_to_image(fft_result, shape=image_size)

        img.save(output_path)
        if verbose:
            print(f"[✓] FFT image saved to: {output_path}")

        return output_path
    except Exception as e:
        print(f"[!] Failed to process {csv_path.name}: {e}")
        return None

# === Standalone usage ===
if __name__ == "__main__":
    from tkinter import filedialog, Tk
    Tk().withdraw()
    selected_csv = filedialog.askopenfilename(
        title="Select a skeleton CSV file",
        filetypes=[("CSV files", "*.csv")]
    )

    if selected_csv:
        csv_to_fft_image(selected_csv)
    else:
        print("❌ No file selected.")
