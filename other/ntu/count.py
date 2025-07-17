import pandas as pd
import numpy as np

# Load CSV (use raw string or forward slashes)
df = pd.read_csv(r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\L\S001C001P001R001A001.skeleton.csv")

# Drop frame index column if it exists
if df.columns[0].lower().startswith("frame") or df.columns[0] == "0":
    df = df.iloc[:, 1:]

# Convert to NumPy
data = df.values

# Find columns that are not all zero
non_zero_columns = np.any(data != 0, axis=0)

# Count them
count = np.sum(non_zero_columns)
print(f"Non-zero columns: {count}")
