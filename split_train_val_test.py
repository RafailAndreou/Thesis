import os
import shutil
import csv
import random

# === CONFIG ===
base_path = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset"
image_dir = os.path.join(base_path, "fft_pku_images")
label_csv = os.path.join(base_path, "labels.csv")
output_dir = os.path.join(base_path, "train_dataset")

split_ratio = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

# === Step 1: Load labels.csv
data = []
with open(label_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append((row["filename"], int(row["label"])))

# === Step 2: Shuffle and Split
random.shuffle(data)
total = len(data)
train_end = int(total * split_ratio["train"])
val_end = train_end + int(total * split_ratio["val"])

splits = {
    "train": data[:train_end],
    "val": data[train_end:val_end],
    "test": data[val_end:]
}

# === Step 3: Create folder structure
for split in splits:
    for i in range(1, 52):  # Actions 1–51
        os.makedirs(os.path.join(output_dir, split, f"{i:02d}"), exist_ok=True)

# === Step 4: Copy images
for split, items in splits.items():
    for fname, label in items:
        src = os.path.join(image_dir, fname)
        dst = os.path.join(output_dir, split, f"{label:02d}", fname)
        shutil.copyfile(src, dst)

print("✅ Dataset created at:", output_dir)
