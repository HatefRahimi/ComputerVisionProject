#############  Part 5 ##############
import numpy as np
import matplotlib.pyplot as plt
import rawpy
from methods import detect_bayer_pattern, create_bayer_masks

# Define file paths and exposure times
data_root = 'exercise_4_data/05'
image_numbers = range(3044, 3050)
exposure_times = np.array([1/10, 1/20, 1/40, 1/80, 1/160, 1/320], dtype=float)

# Load one image to detect pattern and get dimensions
with rawpy.imread(f'{data_root}/IMG_3044.CR3') as rawp:
    raw_sensor = rawp.raw_image_visible.astype(np.float32)

height, width = raw_sensor.shape

# Detect Bayer pattern and create masks
pattern = detect_bayer_pattern(raw_sensor, verbose=True)
red_mask, green_mask, blue_mask = create_bayer_masks((height, width), pattern)

# Compute mean raw values per channel for each exposure
means_red = []
means_green = []
means_blue = []

for num in image_numbers:
    filepath = f'{data_root}/IMG_{num}.CR3'  # Changed 'path' to 'filepath'
    with rawpy.imread(filepath) as rawp:
        raw = rawp.raw_image_visible.astype(np.float32)
    means_red.append(raw[red_mask].mean())
    means_green.append(raw[green_mask].mean())
    means_blue.append(raw[blue_mask].mean())

# Plot with LINEAR scale
plt.figure(figsize=(8, 6))
plt.plot(exposure_times, means_red, '-o', color='r', label='Red')
plt.plot(exposure_times, means_green, '-s', color='g', label='Green')
plt.plot(exposure_times, means_blue, '-^', color='b', label='Blue')

plt.xlabel('Exposure time (s)')
plt.ylabel('Mean raw pixel value')
plt.title('Sensor Linearity: Mean Value vs. Exposure Time')
plt.grid(True, ls='--', lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()
