import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.ndimage import binary_opening, binary_closing
from first import fit_plane

np.random.seed(42)

# Load data
data = loadmat("data/example2kinect.mat")
point_cloud = data['cloud2']
amplitude_image = data['amplitudes2']


def fit_plane_linear(point_one, point_two, point_three):

    points_matrix = np.vstack([point_one, point_two, point_three])
    constants = np.ones(3)

    try:
        normal = np.linalg.solve(points_matrix, constants)
        d = 1.0
        return normal, d
    except np.linalg.LinAlgError:
        return None, None


def mlesac_plane_fit(points, threshold=0.02, max_iterations=1000, gamma=None):
    """MLESAC for plane fitting using linear system method"""
    best_plane = None
    best_cost = float('inf')
    best_inliers = []

    num_points = points.shape[0]

    if gamma is None:
        gamma = threshold ** 2

    for _ in range(max_iterations):
        indices = np.random.choice(num_points, 3, replace=False)
        p0, p1, p2 = points[indices[0]], points[indices[1]], points[indices[2]]

        normal, d = fit_plane(p0, p1, p2)

        if normal is None:
            continue

        distances = np.abs((np.dot(points, normal)) - d)

        costs = np.where(distances < threshold,
                         distances ** 2,
                         gamma)

        total_cost = np.sum(costs)

        if total_cost < best_cost:
            best_cost = total_cost
            best_plane = (normal, d)
            best_inliers = np.where(distances < threshold)[0]

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
