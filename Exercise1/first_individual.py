import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.ndimage import binary_opening, binary_closing

np.random.seed(42)

# Load data
data = loadmat("data/example2kinect.mat")
point_cloud = data['cloud2']
amplitude_image = data['amplitudes2']


def mlesac_plane_fit(points, threshold=1, max_iterations=1000, gamma=None):
    """
    Fit a plane using MLESAC (Maximum Likelihood Estimation SAC)

    Parameters:
    - points: Nx3 array of 3D points
    - threshold: epsilon - distance threshold for inliers
    - max_iterations: maximum iterations
    - gamma: penalty for outliers (if None, uses 3*threshold)

    Returns:
    - best_plane: tuple (normal, d) representing the plane
    - best_inliers: indices of inlier points
    """
    # Set gamma if not provided
    if gamma is None:
        gamma = 3 * threshold

    best_plane = None
    best_inliers = []
    best_cost = float('inf')  # MLESAC minimizes cost

    num_points = points.shape[0]

    for iteration in range(max_iterations):
        # Sample 3 points
        indices = np.random.choice(num_points, 3, replace=False)
        sample = points[indices]

        vec1 = sample[1] - sample[0]
        vec2 = sample[2] - sample[0]
        normal = np.cross(vec1, vec2)

        if np.linalg.norm(normal) == 0:
            continue

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        d = np.dot(normal, sample[0])

        # Calculate distances for all points
        distances = np.abs((points @ normal) - d)

        # MLESAC cost calculation
        # If distance < threshold: cost = distance
        # If distance >= threshold: cost = gamma
        costs = np.where(distances < threshold, distances, gamma)
        total_cost = np.sum(costs)

        # Find inliers
        inliers = np.where(distances < threshold)[0]

        if total_cost < best_cost:
            best_cost = total_cost
            best_inliers = inliers
            best_plane = (normal, d)

    return best_plane, best_inliers


depth_image = point_cloud[:, :, 2]
valid_mask = depth_image > 0
valid_points = point_cloud[valid_mask]

print("Testing MLESAC...")
floor_plane, floor_inliers_idx = mlesac_plane_fit(
    valid_points, threshold=0.02, max_iterations=1000)

print(f"Floor inliers found: {len(floor_inliers_idx)}")

# Visualize
floor_mask = np.zeros(valid_points.shape[0], dtype=np.uint8)
floor_mask[floor_inliers_idx] = 1

floor_image_mask = np.zeros(valid_mask.shape, dtype=np.uint8)
floor_image_mask[valid_mask] = floor_mask

plt.figure()
plt.imshow(floor_image_mask, cmap='gray')
plt.title("Floor Mask (MLESAC)")
plt.axis('off')
plt.show()
