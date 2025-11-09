import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.ndimage import binary_opening, binary_closing
from first import fit_plane, point_to_plane_distance

np.random.seed(42)

# Load data
data = loadmat("data/example2kinect.mat")
point_cloud = data['cloud2']
amplitude_image = data['amplitudes2']


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

        distances = point_to_plane_distance(points, normal, d)

        costs = np.where(distances < threshold,
                         distances ** 2,
                         gamma)

        total_cost = np.sum(costs)

        if total_cost < best_cost:
            best_cost = total_cost
            best_plane = (normal, d)
            best_inliers = np.where(distances < threshold)[0]

    return best_plane, best_inliers


def preemptive_ransac(points, M=1000, B=500, threshold=0.02):
    """
    Preemptive RANSAC with preemption function: f(i) = floor(M * 2^(-floor(i/B)))
    """
    num_points = points.shape[0]

    hypotheses = []
    for _ in range(M):
        indices = np.random.choice(num_points, 3, replace=False)
        sample = points[indices]
        normal, d = fit_plane(sample[0], sample[1], sample[2])

        if normal is not None:
            hypotheses.append({'plane': (normal, d), 'cost': 0.0})

    if len(hypotheses) == 0:
        return None, []

    # Shuffle evaluation order
    eval_indices = np.random.permutation(num_points)
    current_hypotheses = hypotheses
    evaluated_points = 0

    # Iteratively evaluate and prune
    while len(current_hypotheses) > 1 and evaluated_points < num_points:
        # Preemption function
        keep_count = max(1, int(M * (2 ** (-int(evaluated_points / B)))))

        # Evaluate on block
        points_to_eval = min(B, num_points - evaluated_points)
        eval_slice = eval_indices[evaluated_points:
                                  evaluated_points + points_to_eval]

        # Score all hypotheses on this block
        for hyp in current_hypotheses:
            normal, d = hyp['plane']
            block_points = points[eval_slice]
            distances = point_to_plane_distance(block_points, normal, d)
            inliers = np.sum(distances < threshold)
            hyp['cost'] -= inliers  # Negative because we want max inliers

        # Sort and keep best
        current_hypotheses = sorted(
            current_hypotheses, key=lambda h: h['cost'])
        current_hypotheses = current_hypotheses[:keep_count]
        evaluated_points += points_to_eval

    # Get final inliers
    best_hypothesis = current_hypotheses[0]
    normal, d = best_hypothesis['plane']
    distances = point_to_plane_distance(points, normal, d)
    best_inliers = np.where(distances < threshold)[0]

    return best_hypothesis['plane'], best_inliers


depth_image = point_cloud[:, :, 2]
valid_mask = depth_image > 0
valid_points = point_cloud[valid_mask]

# Detect floor with MLESAC
print("Detecting floor with MLESAC...")
floor_plane, best_inliers = mlesac_plane_fit(
    valid_points, threshold=0.02, max_iterations=1000)

print(f"Floor inliers found: {len(best_inliers)}")

# Visualize floor mask
floor_mask = np.zeros(valid_points.shape[0], dtype=np.uint8)
floor_mask[best_inliers] = 1

floor_image_mask = np.zeros(valid_mask.shape, dtype=np.uint8)
floor_image_mask[valid_mask] = floor_mask

# Morphological filtering on floor
floor_clean = binary_closing(floor_image_mask, structure=np.ones((5, 5)))
floor_clean = binary_opening(floor_clean, structure=np.ones((3, 3)))

plt.figure()
plt.imshow(floor_clean, cmap='gray')
plt.title("Floor Mask (MLESAC)")
plt.axis('off')
plt.show()

# Detect box top with Preemptive RANSAC
print("\nDetecting box top with Preemptive RANSAC...")

# Extract non-floor points
floor_bool = floor_clean.astype(bool)
non_floor_mask = valid_mask & (~floor_bool)
non_floor_points = point_cloud[non_floor_mask]

print(f"Non-floor points: {len(non_floor_points)}")

# Preemptive RANSAC
box_plane, box_inliers_idx = preemptive_ransac(
    non_floor_points, M=10, B=10, threshold=0.01)

print(f"Box top inliers found: {len(box_inliers_idx)}")

# Create box mask
box_mask = np.zeros(non_floor_points.shape[0], dtype=np.uint8)
box_mask[box_inliers_idx] = 1

box_image_mask = np.zeros(valid_mask.shape, dtype=np.uint8)
box_image_mask[non_floor_mask] = box_mask

# Find largest connected component
labeled_mask, num_labels = label(box_image_mask)
component_sizes = np.bincount(labeled_mask.ravel())[1:]
largest_component = np.argmax(component_sizes) + 1
box_top_mask = (labeled_mask == largest_component).astype(np.uint8)

plt.figure()
plt.imshow(box_top_mask, cmap='gray')
plt.title("Box Top Mask (Preemptive RANSAC)")
plt.axis('off')
plt.show()

# Calculate box dimensions
print("\nStep 3: Calculating box dimensions...")

normal_floor, d_floor = floor_plane
normal_box, d_box = box_plane

# Height: distance between two parallel planes
height = np.abs(d_box - d_floor)

# Get box top points
box_top_points = point_cloud[box_top_mask == 1]

# Length and width: min/max of x and y coordinates
min_x, max_x = np.min(box_top_points[:, 0]), np.max(box_top_points[:, 0])
min_y, max_y = np.min(box_top_points[:, 1]), np.max(box_top_points[:, 1])

length = np.abs(max_x - min_x)
width = np.abs(max_y - min_y)

print(f"\nBox Dimensions:")
print(f"  Height: {height:.4f} meters")
print(f"  Length: {length:.4f} meters")
print(f"  Width:  {width:.4f} meters")

# Final overlay visualization
plt.figure(figsize=(10, 8))
plt.imshow(depth_image, cmap='plasma')
plt.title('Box Detection Result')
plt.colorbar(label='Distance from Camera (meters)')
plt.imshow(floor_clean, cmap='Greens', alpha=0.3)
plt.imshow(box_top_mask, cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()
