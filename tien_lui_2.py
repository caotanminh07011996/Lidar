import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def dtw_2d(point_cloud1, point_cloud2):
    n = len(point_cloud1)
    m = len(point_cloud2)
    
    D = np.zeros((n + 1, m + 1))
    D[0, 1:] = float('inf')
    D[1:, 0] = float('inf')

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean_distance(point_cloud1[i-1], point_cloud2[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    i, j = n, m
    correspondences = []
    while i > 0 and j > 0:
        correspondences.append((i-1, j-1))
        min_cost = min(D[i-1, j], D[i, j-1], D[i-1, j-1])
        if min_cost == D[i-1, j]:
            i -= 1
        elif min_cost == D[i, j-1]:
            j -= 1
        else:
            i -= 1
            j -= 1

    correspondences.reverse()
    return correspondences

def best_fit_transform(A, B):
    assert A.shape == B.shape

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)

    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)

    return R, t

def icp(A, B, max_iterations=20, tolerance=1e-5):
    P_transformed = P.copy()
    A = np.array(A)
    B = np.array(B)
    prev_error = float('inf')

    for i in range(max_iterations):
        correspondences = dtw_2d(A, B)
        A_corr = np.array([A[i] for i, j in correspondences])
        B_corr = np.array([B[j] for i, j in correspondences])

        R, t = best_fit_transform(A_corr, B_corr)

        A_trans = np.dot(R, A.T).T + t.T

        mean_error = np.mean(np.linalg.norm(A_trans - B_corr, axis=1))

        if abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error
    theta = np.arctan2(R[1,0],R[0, 0])
    angle = np.rad2deg(theta)
    print("Rotation matrix R:")
    print(R)
    print("Theta angle:")
    print(angle)
    print("Translation vector t:")
    print(t)

    return A_trans, R, t


file_path_1 = '/home/tanminh/Documents/SLAM_Lidar_RCT/data2406/Output/frame_2.csv'
file_path_2 = '/home/tanminh/Documents/SLAM_Lidar_RCT/data2406/Output/frame_300.csv'


df1 = pd.read_csv(file_path_1)
P = df1[['X', 'Y']].values.astype(np.float32)
df2 = pd.read_csv(file_path_2)
Q = df2[['X', 'Y']].values.astype(np.float32)


P_trans, R, t = icp(P, Q)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(Q[:, 0], Q[:, 1], color='blue', label='Frame T0')
ax.scatter(P[:, 0], P[:, 1], color='red', label='Frame T1 meansure')
ax.scatter(P_trans[:, 0], P_trans[:, 1], color='green', label='Frame T1 estimate')
ax.legend()
ax.set_title('ICP Alignment for LIDAR 2D Scans')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()

