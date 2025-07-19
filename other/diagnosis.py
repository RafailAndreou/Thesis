import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Your parameters
num = 159
target_cols = 150

def debug_pipeline_step_by_step(csv_file_path):
    """Debug each step of your preprocessing pipeline"""
    
    print(f"🔬 DEBUGGING PIPELINE FOR: {os.path.basename(csv_file_path)}")
    print("=" * 80)
    
    # Step 1: Read raw CSV exactly as you do
    print("\n📄 STEP 1: Reading raw CSV...")
    raw_data = []
    with open(csv_file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        row_count = 0
        for row in reader:
            if row_count < 5:  # Show first 5 rows
                print(f"Raw row {row_count}: {row[:10]}...")  # Show first 10 values
            raw_data.append(row)
            row_count += 1
    
    print(f"Raw data: {len(raw_data)} rows, avg {np.mean([len(row) for row in raw_data]):.1f} cols per row")
    
    # Step 2: Apply your cleaning logic
    print("\n🧹 STEP 2: Applying your cleaning logic...")
    cleaned_data = []
    for i, row in enumerate(raw_data):
        # Your exact cleaning logic
        row_cleaned = [val.strip() for val in row if val.strip() != '']
        if i < 3:  # Show first 3 cleaned rows
            print(f"Cleaned row {i}: {row_cleaned[:10]}...")
        cleaned_data.append(row_cleaned)
    
    print(f"Cleaned data: {len(cleaned_data)} rows, avg {np.mean([len(row) for row in cleaned_data]):.1f} cols per row")
    
    # Step 3: Convert to float and pad
    print("\n🔢 STEP 3: Converting to float and padding...")
    processed_data = []
    for i, row in enumerate(cleaned_data):
        try:
            row_vals = [float(val) for val in row]
            if len(row_vals) >= 75:
                # Your exact padding logic
                padded = row_vals[:target_cols] + [0.0] * max(0, target_cols - len(row_vals))
                processed_data.append(padded)
                
                if i < 3:  # Show first 3 processed rows
                    print(f"Processed row {i}: len={len(padded)}, range=[{min(padded):.3f}, {max(padded):.3f}]")
                    print(f"  First 10 values: {padded[:10]}")
                    print(f"  Non-zero count: {np.count_nonzero(padded)}/{len(padded)} ({np.count_nonzero(padded)/len(padded)*100:.1f}%)")
        except ValueError as e:
            print(f"❌ Error in row {i}: {e}")
            continue
    
    if len(processed_data) == 0:
        print("❌ No valid data after processing!")
        return
    
    signal_img = np.array(processed_data)
    print(f"Signal image shape: {signal_img.shape}")
    print(f"Signal image range: [{np.min(signal_img):.3f}, {np.max(signal_img):.3f}]")
    print(f"Signal image zeros: {(signal_img == 0).sum()}/{signal_img.size} ({(signal_img == 0).sum()/signal_img.size*100:.1f}%)")
    
    # Step 4: Transpose
    print("\n🔄 STEP 4: Transpose to (joints, time)...")
    c = signal_img.T
    print(f"Transposed shape: {c.shape}")
    print(f"First joint stats: min={np.min(c[0]):.3f}, max={np.max(c[0]):.3f}, std={np.std(c[0]):.3f}")
    
    # Step 5: Interpolation
    print("\n🎯 STEP 5: Interpolation...")
    interpolated = []
    for joint_idx in range(min(3, c.shape[0])):  # Check first 3 joints
        joint_series = c[joint_idx]
        print(f"Joint {joint_idx} before interpolation: len={len(joint_series)}, range=[{np.min(joint_series):.3f}, {np.max(joint_series):.3f}]")
        
        f_interp = interpolate.interp1d(
            np.arange(len(joint_series)),
            joint_series,
            kind='linear',
            fill_value="extrapolate"
        )
        interpolated_series = f_interp(np.linspace(0, len(joint_series) - 1, num))
        interpolated.append(interpolated_series)
        
        print(f"Joint {joint_idx} after interpolation: len={len(interpolated_series)}, range=[{np.min(interpolated_series):.3f}, {np.max(interpolated_series):.3f}]")
    
    # Complete interpolation for all joints
    interpolated_full = []
    for joint_series in c:
        f_interp = interpolate.interp1d(
            np.arange(len(joint_series)),
            joint_series,
            kind='linear',
            fill_value="extrapolate"
        )
        interpolated_full.append(f_interp(np.linspace(0, len(joint_series) - 1, num)))
    
    znew = np.array(interpolated_full).T
    print(f"Final interpolated shape: {znew.shape}")
    print(f"Final interpolated range: [{np.min(znew):.3f}, {np.max(znew):.3f}]")
    print(f"Final interpolated zeros: {(znew == 0).sum()}/{znew.size} ({(znew == 0).sum()/znew.size*100:.1f}%)")
    
    # Step 6: Visualize key joints
    print("\n📊 STEP 6: Visualizing joint trajectories...")
    plt.figure(figsize=(15, 10))
    
    # Plot original vs interpolated for first 3 joints
    for i in range(min(3, c.shape[0])):
        plt.subplot(2, 3, i+1)
        plt.plot(c[i], 'b-', alpha=0.7, label='Original')
        plt.title(f'Joint {i} - Original (len={len(c[i])})')
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.legend()
        
        plt.subplot(2, 3, i+4)
        plt.plot(znew[:, i], 'r-', alpha=0.7, label='Interpolated')
        plt.title(f'Joint {i} - Interpolated (len={len(znew[:, i])})')
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Step 7: Check for data corruption patterns
    print("\n🔍 STEP 7: Checking for data corruption patterns...")
    
    # Check if interpolation is flattening data
    original_std = np.std(c[0])
    interpolated_std = np.std(znew[:, 0])
    print(f"Joint 0 std - Original: {original_std:.6f}, Interpolated: {interpolated_std:.6f}")
    
    if interpolated_std < original_std * 0.1:
        print("⚠️  WARNING: Interpolation is severely flattening the data!")
    
    # Check for padding issues
    zero_columns = []
    for i in range(znew.shape[1]):
        if np.all(znew[:, i] == 0):
            zero_columns.append(i)
    
    if len(zero_columns) > 0:
        print(f"⚠️  WARNING: {len(zero_columns)} columns are all zeros: {zero_columns[:10]}...")
    
    # Return data for further analysis
    return {
        'raw_data': raw_data,
        'signal_img': signal_img,
        'interpolated': znew,
        'zero_columns': zero_columns
    }

def compare_multiple_files(file_paths):
    """Compare processing results across multiple files"""
    print(f"\n🔄 COMPARING MULTIPLE FILES")
    print("=" * 50)
    
    results = []
    for file_path in file_paths:
        print(f"\n>>> Processing {os.path.basename(file_path)}")
        try:
            result = debug_pipeline_step_by_step(file_path)
            if result:
                results.append({
                    'filename': os.path.basename(file_path),
                    'result': result
                })
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    # Compare results
    if len(results) >= 2:
        print(f"\n📊 COMPARISON SUMMARY:")
        for i, r in enumerate(results):
            data = r['result']['interpolated']
            print(f"{r['filename']}: shape={data.shape}, range=[{np.min(data):.3f}, {np.max(data):.3f}], zeros={((data == 0).sum()/data.size*100):.1f}%")

# Usage example:
if __name__ == "__main__":
    # Replace with your actual file paths
    test_files = [
        r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\L\S017C001P020R002A053.skeleton.csv",
        r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\L\S017C001P020R002A051.skeleton.csv"
    ]
    
    print("🚀 PIPELINE DEBUGGING SCRIPT")
    print("This will show you exactly where your data gets corrupted!")
    print("\n1. Run debug_pipeline_step_by_step() on a single file")
    print("2. Look for where the data becomes flat/static")
    print("3. Compare multiple files to see consistency")
    
    # Example single file debug:
    # debug_pipeline_step_by_step("your_file.csv")
    
    # Example multiple file comparison:
    # compare_multiple_files(test_files)