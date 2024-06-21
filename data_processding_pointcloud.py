import pandas as pd
import numpy as np
import os
import csv
import math

print("Data processing script")
def convert_to_point_cloud(angle, distance):
    
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)
    return x, y
# Read Input data ###

input_data = "LIDAR_data.xlsx"
output_dir = 'output'
#     Save dataset
with open('Data_raw.xls', 'r') as file:
    # reader = csv.reader(file)
    frame_index = 0
    for line in file:
        # Print the line
        points = []
        if line.startswith('LIDAR'):
            line_list = line.split()
            for i in range(1, len(line_list)-1, 2):
                angle = float(line_list[i].replace(',', '.'))
                distance = float(line_list[i + 1].replace(',', ''))
                if math.isnan(distance) or math.isnan(angle):
                    continue
                x, y = convert_to_point_cloud(angle, distance)
                points.append((x, y))

            frame_index += 1
            point_cloud_data = pd.DataFrame(points, columns=['X', 'Y']) 
            output_file = os.path.join(output_dir, f'frame_{frame_index + 1}.csv')
            point_cloud_data.to_csv(output_file, index=False)



