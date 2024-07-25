import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import csv
from piv_lib import piv  # Importing the PIV function from piv_lib.py

# %% Setting the PIV parameters:
iw = 51  # Interrogation Window Size (pixel)
sw = 81  # Search Window Size (sw > iw) (pixel)

r_limit = 0.5  # Minimum acceptable correlation coefficient. If you're not sure, start with 0.6
i_fix = 500  # Number of maximum correction cycles; 0 means no correction
l_scale = 0.0016  # Spatial scale [m/pixel]; 1 means no size scaling
t_scale = 0.033  # Time step = 1/frame_rate [s/frame]; 1 means no time scaling
cores = -1  # Number of parallel processes: 1 = no parallel processing; 2 and above = number of parallel processes; -1 = maximum

# %% Folder containing the images:
image_folder = '/home/shivanshsinghSUREintern2024/iit_bhu/output_images'

# %% Get a list of all image files in the folder:
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])

# %% Initialize the CSV file:
csv_file = '/home/shivanshsinghSUREintern2024/iit_bhu/results.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename1', 'filename2', 'max_velocity'])

# %% Loop through the image pairs and process:
for i in range(len(image_files) - 1):
    img_1_path = os.path.join(image_folder, image_files[i])
    img_2_path = os.path.join(image_folder, image_files[i + 1])
    
    img_1 = (np.flip(cv2.imread(img_1_path, 0), 0)).astype('float32')  # Read Grayscale
    img_2 = (np.flip(cv2.imread(img_2_path, 0), 0)).astype('float32')

    # %% Running PIV function:
    X, Y, vecx, vecy, vec, rij = piv(img_1, img_2, iw, sw, r_limit, i_fix, l_scale, t_scale, cores)

    # %% Calculate the maximum velocity:
    velocity_magnitude = np.sqrt(vecx**2 + vecy**2)
    max_velocity = np.max(velocity_magnitude)
    
    # %% Append the results to the CSV file:
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_files[i], image_files[i + 1], max_velocity])