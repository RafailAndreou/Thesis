import os
import pandas as pd
import numpy as np

# Paths to scan and output
input_base = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET"
output_base = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\nozero"
subfolders = ['L', 'M', 'R']

# Ensure output directory exists
os.makedirs(output_base, exist_ok=True)

# Iterate through each subfolder
for sub in subfolders:
    input_folder = os.path.join(input_base, sub)
    output_folder = os.path.join(output_base, sub)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                df = pd.read_csv(input_path, header=None)
                # Remove columns where all values are zero
                df_no_zeros = df.loc[:, ~(df == 0).all()]
                df_no_zeros.to_csv(output_path, header=False, index=False)

                print(f"{filename}: {df_no_zeros.shape[1]} columns after zero-removal")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
