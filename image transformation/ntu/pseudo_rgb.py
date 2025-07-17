"""
This script converts a single action CSV file into a pseudo-RGB image.
"""

import numpy as np
import os
from scipy import interpolate
import imageio

# Parameters
joints = 25
joints3D = 75
num = 150

# File paths
csv_file_path = r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\M\S001C002P001R001A041.skeleton.csv'
output_dir = r'C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\pseudo-rgb'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_pseudo_rgb_image(csv_path, output_path):
    """
    Converts a single CSV file into a pseudo-RGB image
    """
    # Parameters for normalization
    minv = [0, 0, 0]
    maxv = [0, 0, 0]
    nmax = 0
    oldmax = 0.1
    oldmin = (-0.2)
    newmax = 255.0
    newmin = 0.0
    oldrange = (oldmax - oldmin)
    newrange = (newmax - newmin)
    
    # Check if file is not empty
    if not os.stat(csv_path).st_size == 0:
        ActionFile = np.genfromtxt(csv_path, delimiter=',')
        
        try:
            xaction, yaction = np.shape(ActionFile)
        except:
            xaction = 1
            yaction = 150
        
        # red is x, green y, blue z
        if xaction != 0:
            if nmax < xaction:
                nmax = xaction

            shape1 = (xaction, joints)
            
            red_pos = np.ndarray(shape1)
            red_pos = np.zeros(shape1)

            green_pos = np.ndarray(shape1)
            green_pos = np.zeros(shape1)

            blue_pos = np.ndarray(shape1)
            blue_pos = np.zeros(shape1)
            
            # Extract RGB positions from the action file
            for j in range(0, xaction):
                y = 0
                x = 0
                while y < joints3D:
                    red_pos[j, x] = ActionFile[j][y]
                    green_pos[j, x] = ActionFile[j][y + 1]
                    blue_pos[j, x] = ActionFile[j][y + 2]
                    y = y + 3
                    x = x + 1
            
            # Initialize arrays for interpolation
            zr = []
            zg = []
            zb = []
            br = []
            bg = []
            bb = []
            
            # Transpose positions
            TransposeRedPos = red_pos.T
            TransposegreenPos = green_pos.T
            TransposeBluePos = blue_pos.T
            
            if xaction >= num:
                dif = num
            else:
                dif = num - xaction
            
            shape = (joints, dif)
            dr = np.zeros(shape)
            dg = np.zeros(shape)
            db = np.zeros(shape)
            
            # Interpolation
            for k in range(0, joints):
                arr2Red = np.array(TransposeRedPos[k])
                arr2Green = np.array(TransposegreenPos[k])
                arr2Blue = np.array(TransposeBluePos[k])

                arr2_interpRed = interpolate.interp1d(np.arange(arr2Red.size), arr2Red)
                arr2_interpGreen = interpolate.interp1d(np.arange(arr2Green.size), arr2Green)
                arr2_interpBlue = interpolate.interp1d(np.arange(arr2Blue.size), arr2Blue)

                arr2_stretchRed = arr2_interpRed(np.linspace(0, arr2Red.size - 1, num))
                arr2_stretchGreen = arr2_interpGreen(np.linspace(0, arr2Green.size - 1, num))
                arr2_stretchBlue = arr2_interpBlue(np.linspace(0, arr2Blue.size - 1, num))

                br = np.concatenate((br, arr2_stretchRed), axis=0)
                bg = np.concatenate((bg, arr2_stretchGreen), axis=0)
                bb = np.concatenate((bb, arr2_stretchBlue), axis=0)
            
            zr = np.reshape(br, (joints, num))
            zb = np.reshape(bg, (joints, num))
            zg = np.reshape(bb, (joints, num))
            zr = zr.T
            zg = zg.T
            zb = zb.T
            
            # Create RGB image
            rgb = np.zeros((joints, 149, 3), dtype=np.uint8)
            redcord = np.zeros((joints, 149, 1), dtype=np.uint8)
            greencord = np.zeros((joints, 149, 1), dtype=np.uint8)
            bluecord = np.zeros((joints, 149, 1), dtype=np.uint8)
            red = np.zeros((joints, 149, 1))
            green = np.zeros((joints, 149, 1))
            blue = np.zeros((joints, 149, 1))
            
            for i in range(0, 149):
                # for every frame
                y = 0
                x = 0

                while y < joints3D:
                    red[x, i] = (zr[i + 1, x] - zr[i, x])
                    green[x, i] = (zg[i + 1, x] - zg[i, x])
                    blue[x, i] = (zb[i + 1, x] - zb[i, x])

                    # Clamp values
                    if red[x, i] > 0.1:
                        red[x, i] = 0.1
                    if green[x, i] > 0.1:
                        green[x, i] = 0.1
                    if blue[x, i] > 0.1:
                        blue[x, i] = 0.1

                    if red[x, i] < -0.2:
                        red[x, i] = -0.2
                    if green[x, i] < -0.2:
                        green[x, i] = -0.2
                    if blue[x, i] < -0.2:
                        blue[x, i] = -0.2

                    # Normalize to 0-255 range
                    redcord[x, i] = ((red[x, i] - oldmin) / (oldmax - oldmin)) * 255.0
                    greencord[x, i] = ((green[x, i] - oldmin) / (oldmax - oldmin)) * 255.0
                    bluecord[x, i] = ((blue[x, i] - oldmin) / (oldmax - oldmin)) * 255.0

                    rgb[x, i][0] = redcord[x, i]  # red
                    rgb[x, i][1] = greencord[x, i]  # green
                    rgb[x, i][2] = bluecord[x, i]  # blue

                    y = y + 3
                    x = x + 1
            
            # Save the RGB image
            imageio.imwrite(output_path, rgb)
            print(f'Saved pseudo-RGB image: {output_path}')

# Process the single CSV file
base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
output_filename = os.path.join(output_dir, base_filename + '.png')

print(f'Processing: {csv_file_path}')
create_pseudo_rgb_image(csv_file_path, output_filename)
print('Processing complete!')