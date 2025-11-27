import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.ndimage import binary_opening, binary_closing
from planes import fit_plane, point_to_plane_distance


np.random.seed(42)

data = loadmat("data/example3kinect.mat")
point_cloud = data['cloud3']


amplitude_image = data['amplitudes3']
plt.figure()
plt.imshow(amplitude_image, cmap='gray')
plt.title("Amplitude Image (A)")
plt.axis('off')
plt.show()

pc = point_cloud.reshape(-1, 3)
n_samples = min(10000, pc.shape[0])
idx = np.random.choice(pc.shape[0], n_samples, replace=False)
pc_samp = pc[idx]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pc_samp[:, 0], pc_samp[:, 1], pc_samp[:, 2], s=1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("3D Scatter of Point Cloud PC")
plt.show()


def ransac_plane_fit(points, threshold=0.02, max_iterations=1000):
    best_plane = None
    best_inliers = []

    num_points = points.shape[0]

    for _ in range(max_iterations):
        indices = np.random.choice(num_points, 3, replace=False)

        # This sample is a small array containing exactly 3 points, each with 3 coordinates
        # It is a sample plane
        sample = points[indices]

        normal, d = fit_plane(sample[0], sample[1], sample[2])

        if normal is None:
            continue
        normal = normal / np.linalg.norm(normal)

        d = np.dot(normal, sample[0])
        # computes dot products for all points at once.
        distances = point_to_plane_distance(points, normal, d)
        inliers = np.where(distances < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)

    return best_plane, best_inliers


# Selects only z from the point cloud (depth at each pixel)
depth_image = point_cloud[:, :, 2]
valid_mask = depth_image > 0
valid_points = point_cloud[valid_mask]

# Visualize Depth Image
plt.imshow(depth_image, cmap='plasma')
plt.title("Depth Image (Z-Values)")
plt.colorbar(label='Distance from Camera (meters)')
plt.show()

# For floor detection
floor_plane, floor_inliers_idx = ransac_plane_fit(
    valid_points, threshold=0.02, max_iterations=1000)

if len(floor_inliers_idx) == 0:
    print("No floor detected.")

# Create a binary mask for the floor points
floor_mask = np.zeros(valid_points.shape[0], dtype=np.uint8)
floor_mask[floor_inliers_idx] = 1

# 2D image for visuallization
floor_image_mask = np.zeros(valid_mask.shape, dtype=np.uint8)
floor_image_mask[valid_mask] = floor_mask


# Morphological Filter:
floor_clean = binary_closing(floor_image_mask, structure=np.ones((5, 5)))

floor_clean = binary_opening(floor_clean, structure=np.ones((3, 3)))

non_floor_mask = valid_mask & (~floor_clean)
non_floor_points = point_cloud[non_floor_mask]


# Visualize Floor Mask
plt.imshow(floor_clean, cmap='gray')
plt.title("Floor Mask (Detected Floor)")
plt.axis('off')
plt.show()

# Box top
box_plane, box_inliers_idx = ransac_plane_fit(
    non_floor_points, threshold=0.01, max_iterations=500)

box_mask = np.zeros(non_floor_points.shape[0], dtype=np.uint8)
box_mask[box_inliers_idx] = 1


box_image_mask = np.zeros(valid_mask.shape, dtype=np.uint8)
box_image_mask[non_floor_mask] = box_mask


labeled_mask, num_labels = label(box_image_mask)
component_sizes = np.bincount(labeled_mask.ravel())[1:]
largest_component = np.argmax(component_sizes) + 1


box_top_mask = (labeled_mask == largest_component).astype(np.uint8)
plt.imshow(box_top_mask, cmap='gray')
plt.title("Top of Box Mask (Detected Box Top, using cleaned floor removal)")
plt.axis('off')
plt.show()

# Estimating height, width and length
normal_floor, d_floor = floor_plane
normal_box, d_box = box_plane

height = np.abs(d_box - d_floor)

print(f"Box height: {height:.4f} meters")

box_top_points = point_cloud[box_top_mask == 1]

min_x, max_x = np.min(box_top_points[:, 0]), np.max(box_top_points[:, 0])
min_y, max_y = np.min(box_top_points[:, 1]), np.max(box_top_points[:, 1])

length = np.abs(max_x - min_x)
width = np.abs(max_y - min_y)

plt.imshow(depth_image, cmap='plasma')  # Show depth image as background
plt.title("Visualization of Floor, Box, and Box Corners")
plt.colorbar(label='Distance from Camera (meters)')
plt.axis('off')

plt.imshow(floor_clean, cmap='gray', alpha=0.5)

plt.imshow(box_top_mask, cmap='gray', alpha=0.5)

plt.show()

print(f"Box length: {length:.4f} meters")
print(f"Box width:  {width:.4f} meters")
