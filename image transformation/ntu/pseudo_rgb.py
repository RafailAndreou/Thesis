import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

# ==== INPUTS ====
input_root = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\nozero"
output_root = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\pseudo-rgb"
target_shape = (224, 224)  # suitable for ResNet50

os.makedirs(output_root, exist_ok=True)

def normalize_to_uint8(array):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(array)
    return (scaled * 255).astype(np.uint8)

def process_csv(file_path, save_path):
    try:
        # Load CSV (each row = frame, columns = joints * 3)
        data = pd.read_csv(file_path, header=None).to_numpy()
        n_frames, n_cols = data.shape

        # Determine number of joints
        if n_cols == 75:  # 1 person (25 joints * 3 coords)
            n_joints = 25
        elif n_cols == 150:  # 2 people (50 joints * 3 coords)
            n_joints = 50
        else:
            raise ValueError(f"Unexpected number of columns: {n_cols}")

        # Split into X, Y, Z coordinate arrays
        x_coords = data[:, 0::3]
        y_coords = data[:, 1::3]
        z_coords = data[:, 2::3]

        # Normalize each channel separately
        x_img = normalize_to_uint8(x_coords)
        y_img = normalize_to_uint8(y_coords)
        z_img = normalize_to_uint8(z_coords)

        # Stack to form RGB
        pseudo_rgb = np.stack([x_img, y_img, z_img], axis=-1)

        # Resize to match ResNet50 input
        img = Image.fromarray(pseudo_rgb)
        img = img.resize(target_shape, Image.BILINEAR)

        # Save image
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
        print(f"Saved: {save_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# ==== Traverse L, M, R ====
for view in ["L", "M", "R"]:
    input_dir = os.path.join(input_root, view)
    output_dir = os.path.join(output_root, view)
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".png")
            process_csv(input_path, output_path)

print("\n✅ Pseudo-RGB transformation complete!")
print(f"Images saved in: {output_root}")
