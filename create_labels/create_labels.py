from pathlib import Path
import pandas as pd

# === CONFIGURATION ===
fft_root = Path(
    r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\nku\fft_images"
)

# Scan all subfolders (classes like A006, A022, ...)
class_folders = sorted([f.name for f in fft_root.iterdir() if f.is_dir()])
label_map = {cls: idx for idx, cls in enumerate(class_folders)}

# Save as CSV
df = pd.DataFrame(list(label_map.items()), columns=["class_name", "label"])
df.to_csv(fft_root / "label_map.csv", index=False)

print("✅ Label map created:")
print(df)
