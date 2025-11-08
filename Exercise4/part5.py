#############  Part 5 ##############
import numpy as np
import matplotlib.pyplot as plt
import rawpy

# Define file paths and exposure times
folder = 'exercise_4_data/05'
file_numbers = range(3044, 3050)  # 3044, 3045, ..., 3049
times = np.array([1 / 10, 1 / 20, 1 / 40, 1 / 80, 1 / 160, 1 / 320], dtype=float)

# Load one image to get dimensions and build Bayer masks (pattern: G R; B G)
with rawpy.imread(f'{folder}/IMG_3044.CR3') as rawp:
    raw0 = rawp.raw_image_visible.astype(np.float32)
H, W = raw0.shape

# Create masks based on the discovered pattern
r_mask = np.zeros((H, W), bool)
g_mask = np.zeros((H, W), bool)
b_mask = np.zeros((H, W), bool)

g_mask[0::2, 0::2] = True  # Green at (0,0)
g_mask[1::2, 1::2] = True  # and (1,1)
r_mask[0::2, 1::2] = True  # Red at (0,1)
b_mask[1::2, 0::2] = True  # Blue at (1,0)

# Compute mean raw values per channel for each exposure
means_R, means_G, means_B = [], [], []

for num, t in zip(file_numbers, times):
    path = f'{folder}/IMG_{num}.CR3'
    with rawpy.imread(path) as rawp:
        raw = rawp.raw_image_visible.astype(np.float32)
    means_R.append(raw[r_mask].mean())
    means_G.append(raw[g_mask].mean())
    means_B.append(raw[b_mask].mean())

plt.figure(figsize=(8, 6))
plt.loglog(times, means_R, '-o', color='r', label='Red')  # red channel in red
plt.loglog(times, means_G, '-s', color='g', label='Green')  # green channel in green
plt.loglog(times, means_B, '-^', color='b', label='Blue')  # blue channel in blue

plt.xlabel('Exposure time (s)')
plt.ylabel('Mean raw pixel value')
plt.title('Sensor Linearity (Mean Value vs. Exposure Time)')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.show()

#############  Part 5 ##############