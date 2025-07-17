import os,sys,io,shutil,csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate

# Parameters
num = 159
fs = 12.0
lowcut = 1
highcut = 4.5

# Input directories
input_dirs = [
    r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\M',
    r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\L',
    r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\R'
]

# Output directory
output_dir = r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\fft_images'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to read CSV file
def readcsv(filename):
    ifile = open(filename, 'rU')
    reader = csv.reader(ifile, delimiter=',')
    rownum = 0
    a = []
    for row in reader:
        a.append(row)
        rownum += 1
    ifile.close()
    return a

# Process single CSV file
def process_csv(csv_filename):
    print('Processing:', csv_filename)
    
    try:
        # Read the CSV file
        picture = readcsv(csv_filename)
        
        # Get signal shape
        signal_shape = len(picture)
        shape1 = (signal_shape, 75)
        signal_img = np.ndarray(shape1)
        signal_img = np.zeros(shape1)
        
        # Fill signal_img with data
        for j in range(0, signal_shape):
            signal_img[j] = picture[j][0:75]
        
        # Transpose
        c = signal_img.T
        
        # Determine difference for interpolation
        if signal_shape >= num:
            dif = num
        else:
            dif = num - signal_shape
        
        shape = (75, dif)
        d = np.zeros(shape)
        z = []
        b = []
        
        # Interpolation
        for k in range(0, 75):
            arr2 = np.array(c[k])
            arr2_interp = interpolate.interp1d(np.arange(arr2.size), arr2)
            arr2_stretch = arr2_interp(np.linspace(0, arr2.size-1, num))
            b = np.concatenate((b, arr2_stretch), axis=0)
        
        z = np.reshape(b, (75, num))
        znew = z.T
        
        # FFT processing
        fft_array = np.fft.fft2(znew)
        fft_shift = np.fft.fftshift(fft_array)
        
        # Calculate magnitude spectrum
        magnitude_spectrum_fft = 20 * np.log(np.abs(fft_shift))
        
        # Save FFT image
        magnitude = magnitude_spectrum_fft * 255
        
        # Extract filename without extension
        base_filename = os.path.splitext(os.path.basename(csv_filename))[0]
        output_filename = os.path.join(output_dir, base_filename + '.png')
        
        matplotlib.pyplot.imsave(output_filename, magnitude)
        print(f'Saved FFT image: {output_filename}')
        
    except Exception as e:
        print(f'Error processing {csv_filename}: {str(e)}')

# Process all CSV files from M, L, R directories
total_files = 0
processed_files = 0

for input_dir in input_dirs:
    print(f'\nProcessing directory: {input_dir}')
    
    if os.path.exists(input_dir):
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        total_files += len(csv_files)
        
        print(f'Found {len(csv_files)} CSV files in {os.path.basename(input_dir)} directory')
        
        for csv_file in csv_files:
            csv_file_path = os.path.join(input_dir, csv_file)
            process_csv(csv_file_path)
            processed_files += 1
    else:
        print(f'Directory not found: {input_dir}')

print(f'\nProcessing complete!')
print(f'Total files found: {total_files}')
print(f'Files processed: {processed_files}')
print(f'Output directory: {output_dir}')