"""
    Reads a CSV file containing skeletal coordinates and plots the skeleton in a 3D graph.
    The CSV file should have the following format:
    - Each line represents a frame of the skeleton.
    - Each frame should have 75 (x, y, z) coordinates, separated by commas.
    - The first line is ignored as it usually contains headers.
    - The first three coordinates of each frame are ignored as they represent the root position.
    - The remaining coordinates are used to plot the skeleton.
    Returns:
    None
"""
import re
import os,sys,io,shutil,csv
from decimal import Decimal
import math as mt
import numpy as np
import scipy.spatial as ss
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
#label path
path = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ntu\NTU_MEDICAL_+_FULL_DATASET\M\S017C002P020R002A039.skeleton.csv"
#label file
file = open(path)
print(f"File opened successfully: {path}")

# Read all lines first to debug
lines = file.readlines()
file.close()

print(f"Total lines in file: {len(lines)}")
if len(lines) > 0:
    print(f"First line: {lines[0][:100]}...")
if len(lines) > 1:
    print(f"Second line: {lines[1][:100]}...")
    print(f"Second line has {len(lines[1].split(','))} values")

# Read all frames
all_frames = []
for line_num, dataLine in enumerate(lines):
    if line_num == 0:  # Skip header line
        continue
    
    if not dataLine.strip():  # Check if line is empty or whitespace
        continue
        
    values = dataLine.split(',')
    num_values = len(values)
    print(f"Line {line_num}: {num_values} values")
    
    # Use the actual number of values instead of expecting exactly 150
    arr = np.zeros([1, num_values])
    
    try:
        for y in range(0, num_values):
            arr[0,y] = Decimal(values[y])
        all_frames.append(arr)
        print(f"Successfully processed line {line_num}")
    except Exception as e:
        print(f"Error processing line {line_num}: {e}")
        continue  # Skip lines that can't be converted to numbers

print(f"Total frames: {len(all_frames)}")

def process_frame(frame_data):
    arr = frame_data
    xline = []
    yline = []
    zline = []
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    
    skeletalCoordinates=[]
    #loop for 1st rotated file
    max_coords = min(78, arr.shape[1])  # Use actual number of coordinates available
    for i in range(3, max_coords, 3):
        if(i<=75 or (i>75 and i+2 < arr.shape[1])):  # Make sure we don't go out of bounds
            p = np.array([arr[0,i-3],arr[0,i-2],arr[0,i-1]])
            w_string1=""
            w_string1 += '<' + str(p[0])+', '+str(p[1])+', '+str(p[2])+', w=1.0>'
            p2 = w_string1
            xline.append(arr[0,i-3])
            
            yline.append(arr[0,i-2])
         
            zline.append(arr[0,i-1])        
            
            p2 = re.findall("-?\d+.\d+",p2)
            skeletalCoordinates.append(p2[0])
            skeletalCoordinates.append(p2[1])
    
    # Only create skeleton connections if we have enough points
    if len(xline) >= 25:  # Need at least 25 points for the skeleton structure
        x = [xline[0],xline[1],xline[20],xline[2],xline[3],xline[2],xline[20],xline[4],xline[5],xline[6],xline[7],xline[21],xline[22],xline[21],xline[7],xline[6],xline[5]
             ,xline[4],xline[20],xline[8],xline[9],xline[10],xline[11],xline[23],xline[24],xline[23],xline[11],xline[10]
             ,xline[9],xline[8],xline[20],xline[1],xline[0],xline[12],xline[13],xline[14],xline[15],xline[14],
             xline[13],xline[12],xline[0],xline[16],xline[17],xline[18],xline[19]]
        y = [yline[0],yline[1],yline[20],yline[2],yline[3],yline[2],yline[20],yline[4],yline[5],yline[6],yline[7],yline[21],yline[22],yline[21],yline[7],yline[6],yline[5]
             ,yline[4],yline[20],yline[8],yline[9],yline[10],yline[11],yline[23],yline[24],yline[23],yline[11],yline[10]
             ,yline[9],yline[8],yline[20],yline[1],yline[0],yline[12],yline[13],yline[14],yline[15],yline[14],
             yline[13],yline[12],yline[0],yline[16],yline[17],yline[18],yline[19]]
        z = [zline[0],zline[1],zline[20],zline[2],zline[3],zline[2],zline[20],zline[4],zline[5],zline[6],zline[7],zline[21],zline[22],zline[21],zline[7],zline[6],zline[5]
             ,zline[4],zline[20],zline[8],zline[9],zline[10],zline[11],zline[23],zline[24],zline[23],zline[11],zline[10]
             ,zline[9],zline[8],zline[20],zline[1],zline[0],zline[12],zline[13],zline[14],zline[15],zline[14],
             zline[13],zline[12],zline[0],zline[16],zline[17],zline[18],zline[19]]
    else:
        # If we don't have enough points, just plot what we have
        x = xline
        y = yline 
        z = zline
    
    return x, y, z

# Set up the figure and axis
fig, ax = plt.subplots()

# Test with first frame to make sure data is working
if all_frames:
    x, y, z = process_frame(all_frames[0])
    print(f"First frame data - x range: {min(x):.3f} to {max(x):.3f}, y range: {min(y):.3f} to {max(y):.3f}")
    print(f"Number of points: {len(x)}")
else:
    print("No frames loaded!")

# Animation function
def animate(frame_num):
    ax.clear()  # Clear the previous frame
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.2, 0.6)
    ax.set_aspect('equal', adjustable='box')
    
    if frame_num < len(all_frames):
        x, y, z = process_frame(all_frames[frame_num])
        ax.plot(x, y, 'o', color='red')
        ax.plot(x, y, color='blue')
        ax.set_title(f'Frame {frame_num + 1}/{len(all_frames)}')
    
    return ax.get_children()

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=len(all_frames), 
                             interval=10, blit=False, repeat=True)

plt.show()

# Optionally save the animation
# anim.save('skeleton_animation.gif', writer='pillow', fps=10)