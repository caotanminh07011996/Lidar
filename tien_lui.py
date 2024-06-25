import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_closest_points(P, Q):
    if len(P) <= len(Q):
        source, target = P, Q
    else:
        source, target = Q, P
    
    closest_points = []
    for p in source:
        distances = np.linalg.norm(target - p, axis=1)
        closest_points.append(target[np.argmin(distances)])
    
    closest_points = np.array(closest_points)
    
    if len(P) <= len(Q):
        return P, closest_points
    else:
        return closest_points, Q

def compute_centroid(X):
    return np.mean(X, axis=0)

def compute_transformation(P, Q):
    centroid_P = compute_centroid(P)
    centroid_Q = compute_centroid(Q)
    
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    W = np.dot(P_centered.T, Q_centered)
    U, _, Vt = np.linalg.svd(W)
    R = np.dot(U, Vt)
    t = centroid_Q - np.dot(R, centroid_P)
    
    return R, t

def apply_transformation(P, R, t):
    return np.dot(P, R.T) + t

def icp(P, Q, max_iterations=100, tolerance=1e-5):
    P_transformed = P.copy()
    prev_error = np.inf
    
    for i in range(max_iterations):
        closest_points, fixed_points = find_closest_points(P_transformed, Q)
        R, t = compute_transformation(fixed_points, closest_points)
        P_transformed = apply_transformation(fixed_points, R, t)
        
        mean_error = np.mean(np.linalg.norm(P_transformed - closest_points, axis=1))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
        theta = np.arctan2(R[1,0],R[0, 0])
    
    return P_transformed, R, t, theta
# Đường dẫn tới các file CSV
file_path_1 = '/home/tanminh/Documents/SLAM_Lidar_RCT/data2406/Output/frame_2.csv'
file_path_2 = '/home/tanminh/Documents/SLAM_Lidar_RCT/data2406/Output/frame_500.csv'

# Đọc dữ liệu từ các file CSV
df1 = pd.read_csv(file_path_1)
P = df1[['X', 'Y']].values.astype(np.float32)
df2 = pd.read_csv(file_path_2)
Q = df2[['X', 'Y']].values.astype(np.float32)

# Áp dụng thuật toán ICP
P_transformed, R, t , theta= icp(P, Q)

# In ra ma trận quay và vector tịnh tiến
print("Rotation matrix R:")
print(R)
print("Theta angle:")
print(theta)
print("Translation vector t:")
print(t)


# Hiển thị kết quả
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(Q[:, 0], Q[:, 1], color='blue', label='Frame T0')
ax.scatter(P[:, 0], P[:, 1], color='red', label='Frame T1 meansure')
ax.scatter(P_transformed[:, 0], P_transformed[:, 1], color='green', label='Frame T1 estimate')
ax.legend()
ax.set_title('ICP Alignment for LIDAR 2D Scans')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()
