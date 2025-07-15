import os
import shutil
import random
from pathlib import Path

# === CONFIGURATION ===
INPUT_DIR = Path(r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\nku\fft_images")
OUTPUT_BASE = INPUT_DIR.parent  # Will create train/, val/, test/ alongside fft_images/
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

# === SPLIT FUNCTION ===
def split_dataset():
    random.seed(RANDOM_SEED)
    classes = [d for d in INPUT_DIR.iterdir() if d.is_dir()]

    for class_dir in classes:
        images = list(class_dir.glob("*.png"))
        random.shuffle(images)

        total = len(images)
        n_train = int(SPLITS["train"] * total)
        n_val = int(SPLITS["val"] * total)

        split_map = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, split_images in split_map.items():
            out_dir = OUTPUT_BASE / split / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy(img_path, out_dir / img_path.name)

if __name__ == "__main__":
    split_dataset()
    print("✅ Dataset split into train/val/test folders.")
