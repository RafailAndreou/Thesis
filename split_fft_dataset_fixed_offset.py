import os
import shutil
import random

# === CONFIG ===
base_dir = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\train_dataset"
label_csv = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\labelsPKU\labelsM.csv"  # global label list (all 21,453 entries)

fft_sources = {
    "L": {
        "dir": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\fftimagesPKU\fftimages_L",
        "offset": 0
    },
    "M": {
        "dir": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\fftimagesPKU\fftimages_M",
        "offset": 7151
    },
    "R": {
        "dir": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\fftimagesPKU\fftimages_R",
        "offset": 14302
    }
}

train_ratio = 0.85

# === CLEAN & REBUILD FOLDER STRUCTURE ===
for split in ["train", "val", "test"]:
    for i in range(1, 52):  # 1 to 51
        os.makedirs(os.path.join(base_dir, split, f"{i:02d}"), exist_ok=True)

# === Load global label list ===
with open(label_csv, "r") as f:
    labels = [int(line.strip()) for line in f if line.strip().isdigit()]

# === Step 1: Collect files from L and R (for training + val) ===
trainval_data = []

for view in ["L", "R"]:
    folder = fft_sources[view]["dir"]
    offset = fft_sources[view]["offset"]

    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")],
                   key=lambda x: int(x.split("_")[1].split(".")[0]))  # numeric sort

    for i, fname in enumerate(files):
        global_idx = i + offset
        if global_idx >= len(labels):
            print(f"⚠️ Skipping {fname} — label index out of bounds")
            continue
        label = labels[global_idx]
        src = os.path.join(folder, fname)
        trainval_data.append((src, label))

random.shuffle(trainval_data)
split_idx = int(len(trainval_data) * train_ratio)
train_set = trainval_data[:split_idx]
val_set = trainval_data[split_idx:]

# === Copy train/val files ===
for src, label in train_set:
    dest = os.path.join(base_dir, "train", f"{label:02d}", os.path.basename(src))
    shutil.copyfile(src, dest)

for src, label in val_set:
    dest = os.path.join(base_dir, "val", f"{label:02d}", os.path.basename(src))
    shutil.copyfile(src, dest)

# === Step 2: Copy M set as test ===
view = "M"
folder = fft_sources[view]["dir"]
offset = fft_sources[view]["offset"]

files = sorted([f for f in os.listdir(folder) if f.endswith(".png")],
               key=lambda x: int(x.split("_")[1].split(".")[0]))

for i, fname in enumerate(files):
    global_idx = i + offset
    if global_idx >= len(labels):
        print(f"⚠️ Skipping {fname} — label index out of bounds")
        continue
    label = labels[global_idx]
    src = os.path.join(folder, fname)
    dest = os.path.join(base_dir, "test", f"{label:02d}", fname)
    shutil.copyfile(src, dest)

print("✅ Fixed labeling complete. Dataset is ready at:", base_dir)
