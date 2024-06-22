import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/home/tanminh/Documents/Uong/caominh/output/frame_2.csv'
df = pd.read_csv(file_path)

# Convert the DataFrame to a list of points
points = df[['X', 'Y']].values.astype(np.float32)

# Reshape the points to fit the contour format required by approxPolyDP
contour = points.reshape((-1, 1, 2))

# Define the epsilon value for approximation accuracy (a percentage of the arc length)
epsilon = 0.02 * cv2.arcLength(contour, True)

# Apply the approxPolyDP function
approx_contour = cv2.approxPolyDP(contour, epsilon, True)

# Convert the result back to a DataFrame for display
approx_df = pd.DataFrame(approx_contour.reshape(-1, 2), columns=['X', 'Y'])

# Plot the original points and the approximated contour
plt.figure(figsize=(10, 6))
plt.plot(df['X'], df['Y'], 'bo-', label='Original Points')
plt.plot(approx_df['X'], approx_df['Y'], 'ro-', label='Approximated Contour')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Points and Approximated Contour')
plt.legend()
plt.show()

# Display the approximated contour DataFrame
print(approx_df.head())