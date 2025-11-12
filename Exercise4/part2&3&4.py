import matplotlib.pyplot as plt
import numpy as np
import rawpy
from scipy.ndimage import convolve
from part1 import detect_bayer_pattern

#############  PART 2: DEMOSAICING ##############


def create_bayer_masks(shape, pattern):

    height, width = shape

    red_mask = np.zeros((height, width), bool)
    green_mask = np.zeros((height, width), bool)
    blue_mask = np.zeros((height, width), bool)

    # Assign based on detected pattern
    red_offset = pattern['red']
    green1_offset = pattern['green1']
    green2_offset = pattern['green2']
    blue_offset = pattern['blue']

    red_mask[red_offset[0]::2, red_offset[1]::2] = True
    green_mask[green1_offset[0]::2, green1_offset[1]::2] = True
    green_mask[green2_offset[0]::2, green2_offset[1]::2] = True
    blue_mask[blue_offset[0]::2, blue_offset[1]::2] = True

    return red_mask, green_mask, blue_mask


with rawpy.imread('exercise_4_data/02/IMG_4782.CR3') as rawp:
    raw_sensor = rawp.raw_image_visible.astype(np.float32)

height, width = raw_sensor.shape
print(f'Loaded raw image: {height} x {width}\n')

# Detect Bayer pattern
pattern = detect_bayer_pattern(raw_sensor)
red_mask, green_mask, blue_mask = create_bayer_masks((height, width), pattern)

# Extract channels
red_channel = np.zeros_like(raw_sensor)
red_channel[red_mask] = raw_sensor[red_mask]
green_channel = np.zeros_like(raw_sensor)
green_channel[green_mask] = raw_sensor[green_mask]
blue_channel = np.zeros_like(raw_sensor)
blue_channel[blue_mask] = raw_sensor[blue_mask]

# Interpolation
kernel = np.ones((3, 3))


def interpolate_missing_values(channel, mask):
    num = convolve(channel, kernel, mode='mirror')
    denom = convolve(mask.astype(np.float32), kernel, mode='mirror')
    return num / np.maximum(denom, 1e-6)


red_interpolated = interpolate_missing_values(red_channel, red_mask)
green_interpolated = interpolate_missing_values(green_channel, green_mask)
blue_interpolated = interpolate_missing_values(blue_channel, blue_mask)

rgb_linear_image = np.stack(
    [red_interpolated, green_interpolated, blue_interpolated], axis=2)

#############  PART 3: GAMMA CORRECTION ##############


def improve_luminosity(rgb_linear_image, p_low=0.01, p_high=99.99, gamma=0.3, alpha=10.0):
    """
    Apply gamma correction and logarithmic curve.
    Returns both in [0,1] range for display.
    """
    # Global percentiles over all channels
    low_percentile = np.percentile(rgb_linear_image, p_low)
    high_percentile = np.percentile(rgb_linear_image, p_high)

    # Normalize to [0,1]
    normalized_image = (rgb_linear_image - low_percentile) / \
        (high_percentile - low_percentile)
    normalized_image = np.clip(normalized_image, 0.0, 1.0)

    # Gamma correction
    gamma_corrected = np.power(normalized_image, gamma)

    # Log curve (alternative)
    log_corrected = np.log1p(alpha * normalized_image) / np.log1p(alpha)

    return gamma_corrected, log_corrected


gamma_image, log_image = improve_luminosity(rgb_linear_image)


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

#############  EXERCISE 4: WHITE BALANCE ##############


def gray_world_white_balance(image_rgb):

    means = image_rgb.mean(axis=(0, 1))
    scale = means[1] / means
    white_balance = image_rgb * scale[np.newaxis, np.newaxis, :]
    return np.clip(white_balance, 0, 1)


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
