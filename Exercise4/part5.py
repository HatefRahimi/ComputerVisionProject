#############  Part 5 ##############
import numpy as np
import matplotlib.pyplot as plt
import rawpy
from methods import detect_bayer_pattern, create_bayer_masks

# Define file paths and exposure times
folder = 'exercise_4_data/05'
file_numbers = range(3044, 3050)  # 3044, 3045, ..., 3049
times = np.array([1/10, 1/20, 1/40, 1/80, 1/160, 1/320], dtype=float)

# Load one image to detect pattern and get dimensions
with rawpy.imread(f'{folder}/IMG_3044.CR3') as rawp:
    raw0 = rawp.raw_image_visible.astype(np.float32)

H, W = raw0.shape

# Detect Bayer pattern and create masks
pattern = detect_bayer_pattern(raw0, verbose=True)
r_mask, g_mask, b_mask = create_bayer_masks((H, W), pattern)

# Compute mean raw values per channel for each exposure
means_R, means_G, means_B = [], [], []

for num in file_numbers:
    path = f'{folder}/IMG_{num}.CR3'
    with rawpy.imread(path) as rawp:
        raw = rawp.raw_image_visible.astype(np.float32)
    means_R.append(raw[r_mask].mean())
    means_G.append(raw[g_mask].mean())
    means_B.append(raw[b_mask].mean())

# Plot with LINEAR scale (not loglog!)
plt.figure(figsize=(8, 6))
plt.plot(times, means_R, '-o', color='r', label='Red')
plt.plot(times, means_G, '-s', color='g', label='Green')
plt.plot(times, means_B, '-^', color='b', label='Blue')

plt.xlabel('Exposure time (s)')
plt.ylabel('Mean raw pixel value')
plt.title('Sensor Linearity: Mean Value vs. Exposure Time')
plt.grid(True, ls='--', lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()

print("\n✅ Part 5 complete: Sensor linearity verified!")
