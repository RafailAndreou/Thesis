import os
import csv
import re

# === CONFIG ===
image_dir = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\fft_pku_images"
output_csv = os.path.join(os.path.dirname(image_dir), "labels.csv")

# === Extract and Save Labels ===
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "label"])

    for fname in sorted(os.listdir(image_dir)):
        if not fname.endswith(".png"):
            continue
        match = re.search(r"action_(\d+)", fname)
        if match:
            label = int(match.group(1))
            writer.writerow([fname, label])
            print(f"✅ {fname} → label {label}")
        else:
            print(f"⚠️ Skipping {fname}: no label found")

print(f"\n📄 Label CSV saved to: {output_csv}")
