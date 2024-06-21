import pandas as pd
import numpy as np
import os


def convert_to_point_cloud(angle, distance):
    
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)

    return x, y
# Read Input data ###

input_data = "LIDAR_data.xlsx"
output_dir = '/home/tanminh/Documents/SLAM_Lidar_RCT/Dataset'
#     Save dataset


lidar_data_raw = pd.read_excel(input_data)
os.makedirs(output_dir, exist_ok=True)



for frame_index, row in lidar_data_raw.iterrows():
    points = []

    for i in range(0, len(row), 2):
        if i + 1 < len(row):
            
            angle = float(row[i])
            distance = float(row[i + 1])

            if str(distance) != 'NaN':
                x, y = convert_to_point_cloud(angle, distance)
                points.append((x, y))

    



    point_cloud_data = pd.DataFrame(points, columns=['X', 'Y']) 

    output_file = os.path.join(output_dir, f'frame_{frame_index + 1}.csv')
    point_cloud_data.to_csv(output_file, index=False)



print("Done")

