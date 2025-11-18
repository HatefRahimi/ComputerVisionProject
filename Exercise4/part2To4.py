import matplotlib.pyplot as plt
import numpy as np
import rawpy
from methods import (
    detect_bayer_pattern,
    create_bayer_masks,
    interpolate_missing_values,
    gamma_correction,
    gray_world_white_balance,
    demosaic,
    detect_bayer_pattern_fixed
)

#############  PART 2: DEMOSAICING ##############

# Load raw image
with rawpy.imread('exercise_4_data/02/IMG_4782.CR3') as rawp:
    raw_sensor = rawp.raw_image_visible.astype(np.float32)

height, width = raw_sensor.shape
print(f'Loaded raw image: {height} x {width}\n')

# Detect Bayer pattern
pattern = detect_bayer_pattern(raw_sensor, verbose=True)
red_mask, green_mask, blue_mask = create_bayer_masks((height, width), pattern)

rgb_linear_image = demosaic(raw_sensor, pattern)

#############  PART 3: GAMMA CORRECTION ##############

# Apply gamma correction
gamma_image = gamma_correction(rgb_linear_image, gamma=0.3)


# log curve

def log_luminosity(rgb_image, p_low=0.01, p_high=99.99, alpha=10.0):
    low_percentile = np.percentile(rgb_image, p_low)
    high_percentile = np.percentile(rgb_image, p_high)
    normalized = (rgb_image - low_percentile) / \
                 (high_percentile - low_percentile)
    normalized = np.clip(normalized, 0.0, 1.0)
    return np.log1p(alpha * normalized) / np.log1p(alpha)


log_image = log_luminosity(rgb_linear_image)

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(np.clip(rgb_linear_image / np.max(rgb_linear_image), 0, 1))
axs[0].set_title('Linear')
axs[0].axis('off')

axs[1].imshow(gamma_image)
axs[1].set_title('Gamma (γ=0.3)')
axs[1].axis('off')

axs[2].imshow(log_image)
axs[2].set_title('Log (α=10)')
axs[2].axis('off')

plt.tight_layout()
plt.show()

#############  PART 4: WHITE BALANCE ##############

# Apply white balance
white_balanced_image = gray_world_white_balance(gamma_image)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(gamma_image)
axes[0].set_title('After Gamma')
axes[0].axis('off')

axes[1].imshow(white_balanced_image)
axes[1].set_title('After White Balance')
axes[1].axis('off')

plt.tight_layout()
plt.show()
