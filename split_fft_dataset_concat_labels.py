import os
import shutil
import random

# === CONFIG ===
base_dir = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\train_dataset"

fft_sources = {
    "L": {
        "dir": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\fftimagesPKU\fftimages_L",
        "label_csv": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\labelsPKU\labelsL.csv"
    },
    "M": {
        "dir": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\fftimagesPKU\fftimages_M",
        "label_csv": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\labelsPKU\labelsM.csv"
    },
    "R": {
        "dir": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\fftimagesPKU\fftimages_R",
        "label_csv": r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Dataset\Dataset\labelsPKU\labelsR.csv"
    }
}

train_ratio = 0.85

# === Step 0: Load and combine all labels
def load_labels(csv_path):
    with open(csv_path, "r") as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]

labels_all = []
for view in ["L", "M", "R"]:
    labels_all += load_labels(fft_sources[view]["label_csv"])

print(f"✅ Loaded {len(labels_all)} total labels.")

# === CLEAN & REBUILD FOLDER STRUCTURE ===
for split in ["train", "val", "test"]:
    for i in range(1, 52):
        os.makedirs(os.path.join(base_dir, split, f"{i:02d}"), exist_ok=True)

# === Step 1: Process all images with global labels
trainval_data = []
test_data = []

for view in ["L", "R", "M"]:
    folder = fft_sources[view]["dir"]

    files = sorted([f for f in os.listdir(folder) if f.startswith("images_") and f.endswith(".png")],
                   key=lambda x: int(x.split("_")[1].split(".")[0]))

    for fname in files:
        idx = int(fname.split("_")[1].split(".")[0])  # images_XXXX.png → XXXX
        if idx >= len(labels_all):
            print(f"⚠️ Skipping {fname} — label index {idx} out of range")
            continue
        label = labels_all[idx]
        src = os.path.join(folder, fname)

        if view == "M":
            test_data.append((src, label))
        else:
            trainval_data.append((src, label))

# === Step 2: Shuffle and split train/val
random.shuffle(trainval_data)
split_idx = int(len(trainval_data) * train_ratio)
train_set = trainval_data[:split_idx]
val_set = trainval_data[split_idx:]

# === Step 3: Copy to folders
def copy_set(data, split):
    for src, label in data:
        dest = os.path.join(base_dir, split, f"{label:02d}", os.path.basename(src))
        shutil.copyfile(src, dest)

copy_set(train_set, "train")
copy_set(val_set, "val")
copy_set(test_data, "test")

print("✅ Dataset fully labeled and saved to:", base_dir)
