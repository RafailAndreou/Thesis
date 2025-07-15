import os
import shutil
import csv
import random

# === CONFIG ===
base_dir = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\train_dataset"
fft_dirs = {
    "L": {
        "images": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\fftimagesPKU\fftimages_L",
        "labels": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\labelsPKU\labelsL.csv"
    },
    "R": {
        "images": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\fftimagesPKU\fftimages_R",
        "labels": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\labelsPKU\labelsR.csv"
    },
    "M": {
        "images": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\fftimagesPKU\fftimages_M",
        "labels": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\labelsPKU\labelsM.csv"
    }
}

train_ratio = 0.85  # 85% train, 15% validation

# === CLEAN & REBUILD FOLDER STRUCTURE ===
for split in ["train", "val", "test"]:
    for i in range(1, 52):  # action classes 1 to 51
        os.makedirs(os.path.join(base_dir, split, f"{i:02d}"), exist_ok=True)

# === Helper: Load labels from .csv ===
def load_labels(csv_path):
    with open(csv_path, "r") as f:
        return [int(line.strip()) for line in f.readlines() if line.strip().isdigit()]

# === Step 1: Collect L + R → shuffle → train/val ===
all_train_data = []

for view in ["L", "R"]:
    image_dir = fft_dirs[view]["images"]
    label_path = fft_dirs[view]["labels"]
    labels = load_labels(label_path)
    files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    for i, fname in enumerate(files):
        if i >= len(labels): continue
        label = labels[i]
        all_train_data.append((os.path.join(image_dir, fname), label))

random.shuffle(all_train_data)
split_idx = int(len(all_train_data) * train_ratio)
train_set = all_train_data[:split_idx]
val_set = all_train_data[split_idx:]

# === Step 2: Copy train/val images ===
for path, label in train_set:
    dest = os.path.join(base_dir, "train", f"{label:02d}", os.path.basename(path))
    shutil.copyfile(path, dest)

for path, label in val_set:
    dest = os.path.join(base_dir, "val", f"{label:02d}", os.path.basename(path))
    shutil.copyfile(path, dest)

# === Step 3: Copy M set as test ===
image_dir = fft_dirs["M"]["images"]
label_path = fft_dirs["M"]["labels"]
labels = load_labels(label_path)
files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

for i, fname in enumerate(files):
    if i >= len(labels): continue
    label = labels[i]
    src = os.path.join(image_dir, fname)
    dest = os.path.join(base_dir, "test", f"{label:02d}", fname)
    shutil.copyfile(src, dest)

print("✅ Done. Dataset prepared at:", base_dir)
