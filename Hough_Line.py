import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_data(file_path):
    return pd.read_csv(file_path)

def create_image_from_points(data, image_size=(1000, 1000)):
    # Normalize points
    x_normalized = (data['X'] - data['X'].min()) / (data['X'].max() - data['X'].min())
    y_normalized = (data['Y'] - data['Y'].min()) / (data['Y'].max() - data['Y'].min())
    
    # Scale to image size
    x_scaled = (x_normalized * (image_size[0] - 1)).astype(int)
    y_scaled = (y_normalized * (image_size[1] - 1)).astype(int)
    
    # Create an image with white background
    image = np.zeros(image_size, dtype=np.uint8)
    image[y_scaled, x_scaled] = 255  # Set points to white
    
    return image

def detect_lines_with_hough(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    return lines

def plot_detected_lines_with_points(data, lines):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['X'], data['Y'], color='gray', alpha=0.5, s=50, label='Point Cloud')

    if lines is not None:
        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            plt.plot([x1, x2], [y1, y2], 'r', label='Detected Line' if idx == 0 else "")
            plt.scatter([x1, x2], [y1, y2], color='green', s=100, label='Endpoints' if idx == 0 else "")


def interested_point(data, lines):
    plt.figure(figsize=(10, 6))
    # Plot the original points
    plt.scatter(data['X'], data['Y'], color='gray', alpha=0.5, s=50, label='Point Cloud')

    if lines is not None:
        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            # Scale points for plotting
            x1_normalized = (x1 / 1000) * (data['X'].max() - data['X'].min()) + data['X'].min()
            y1_normalized = (y1 / 1000) * (data['Y'].max() - data['Y'].min()) + data['Y'].min()
            x2_normalized = (x2 / 1000) * (data['X'].max() - data['X'].min()) + data['X'].min()
            y2_normalized = (y2 / 1000) * (data['Y'].max() - data['Y'].min()) + data['Y'].min()
            # Plot the lines
            plt.plot([x1_normalized, x2_normalized], [y1_normalized, y2_normalized], 'r', label='Detected Line' if idx == 0 else "")
            # Plot the endpoints
            plt.scatter([x1_normalized, x2_normalized], [y1_normalized, y2_normalized], color='green', s=100, label='Endpoints' if idx == 0 else "")

    plt.title('Point Cloud with Detected Lines and Endpoints')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function to plot the results


# Load data
data = read_data('/home/tanminh/Documents/SLAM_Lidar_RCT/data2406/Output/frame_2.csv')

# Create image from points
image = create_image_from_points(data)

# Detect lines
lines = detect_lines_with_hough(image)

# Plot the detected lines and endpoints on the original point cloud
#plot_detected_lines_with_points(data, lines)

interested_point(data, lines)