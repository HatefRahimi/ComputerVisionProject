"""
Reusable functions for demosaicing and image processing
"""
import numpy as np
from scipy.ndimage import convolve


def detect_bayer_pattern(raw_array, verbose=False):
    """
    Detect the Bayer pattern from raw sensor data by analyzing mean values.

    Args:
        raw_array: 2D numpy array of raw sensor values
        verbose: if True, print detection details

    Returns:
        dict with keys:
            - 'red': (row, col) offset tuple
            - 'green1': (row, col) offset tuple  
            - 'green2': (row, col) offset tuple
            - 'blue': (row, col) offset tuple
            - 'pattern_name': string like 'GRGB' or 'RGGB'
    """
    # Compute mean value at each 2×2 Bayer offset
    means = {
        (dy, dx): raw_array[dy::2, dx::2].mean()
        for dy in (0, 1) for dx in (0, 1)
    }

    if verbose:
        print('Mean values at offsets (dy,dx):')
        for (dy, dx), m in means.items():
            print(f'  Offset ({dy},{dx}): {m:.2f}')

    # Sort by mean (highest to lowest)
    sorted_means = sorted(means.items(), key=lambda x: x[1], reverse=True)

    if verbose:
        print('\nSorted by mean (highest to lowest):')
        for (dy, dx), m in sorted_means:
            print(f'  Offset ({dy},{dx}): {m:.2f}')

    # Identify colors (green typically has highest values)
    green1_offset = sorted_means[0][0]
    green2_offset = sorted_means[1][0]
    red_offset = sorted_means[2][0]
    blue_offset = sorted_means[3][0]

    # Determine pattern name
    pattern_map = {
        (0, 0): None, (0, 1): None,
        (1, 0): None, (1, 1): None
    }
    pattern_map[green1_offset] = 'G'
    pattern_map[green2_offset] = 'G'
    pattern_map[red_offset] = 'R'
    pattern_map[blue_offset] = 'B'

    pattern_name = (pattern_map[(0, 0)] + pattern_map[(0, 1)] +
                    pattern_map[(1, 0)] + pattern_map[(1, 1)])

    if verbose:
        print('\n--- Identified Bayer Pattern ---')
        print(f'Pattern: {pattern_name}')
        print(f'Red:     offset {red_offset}')
        print(f'Green 1: offset {green1_offset}')
        print(f'Green 2: offset {green2_offset}')
        print(f'Blue:    offset {blue_offset}')

    return {
        'red': red_offset,
        'green1': green1_offset,
        'green2': green2_offset,
        'blue': blue_offset,
        'pattern_name': pattern_name
    }


def create_bayer_masks(shape, pattern):
    """
    Create binary masks for each color channel based on Bayer pattern.

    Args:
        shape: (height, width) tuple
        pattern: dict from detect_bayer_pattern()

    Returns:
        (red_mask, green_mask, blue_mask) tuple of boolean arrays
    """
    height, width = shape

    red_mask = np.zeros((height, width), dtype=bool)
    green_mask = np.zeros((height, width), dtype=bool)
    blue_mask = np.zeros((height, width), dtype=bool)

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


def interpolate_missing_values(channel, mask, kernel=None):
    """
    Interpolate missing values using weighted averaging.

    Args:
        channel: 2D array with values only at mask positions
        mask: boolean mask indicating known values
        kernel: convolution kernel (default: 3x3 ones)

    Returns:
        Interpolated channel with all values filled
    """
    if kernel is None:
        kernel = np.ones((3, 3))

    num = convolve(channel, kernel, mode='mirror')
    denom = convolve(mask.astype(np.float32), kernel, mode='mirror')
    return num / np.maximum(denom, 1e-6)


def demosaic(raw_data, pattern=None):
    """
    Complete demosaicing pipeline.

    Args:
        raw_data: 2D raw sensor array
        pattern: optional pre-detected pattern dict

    Returns:
        RGB image (H x W x 3) with interpolated channels
    """
    if pattern is None:
        pattern = detect_bayer_pattern(raw_data)

    height, width = raw_data.shape
    red_mask, green_mask, blue_mask = create_bayer_masks(
        (height, width), pattern)

    # Extract channels
    red_channel = np.zeros_like(raw_data, dtype=np.float32)
    red_channel[red_mask] = raw_data[red_mask]

    green_channel = np.zeros_like(raw_data, dtype=np.float32)
    green_channel[green_mask] = raw_data[green_mask]

    blue_channel = np.zeros_like(raw_data, dtype=np.float32)
    blue_channel[blue_mask] = raw_data[blue_mask]

    # Interpolate
    red_interpolated = interpolate_missing_values(red_channel, red_mask)
    green_interpolated = interpolate_missing_values(green_channel, green_mask)
    blue_interpolated = interpolate_missing_values(blue_channel, blue_mask)

    return np.stack([red_interpolated, green_interpolated, blue_interpolated], axis=2)


def improve_luminosity(rgb_image, p_low=0.01, p_high=99.99, gamma=0.3):
    """
    Apply gamma correction with percentile-based normalization.

    Args:
        rgb_image: RGB image array
        p_low: lower percentile for normalization
        p_high: upper percentile for normalization
        gamma: gamma correction exponent

    Returns:
        Gamma-corrected image in [0, 1] range
    """
    low_percentile = np.percentile(rgb_image, p_low)
    high_percentile = np.percentile(rgb_image, p_high)

    normalized = (rgb_image - low_percentile) / \
        (high_percentile - low_percentile)
    normalized = np.clip(normalized, 0.0, 1.0)

    return np.power(normalized, gamma)


def gray_world_white_balance(rgb_image):
    """
    Apply gray world white balance assumption.

    Args:
        rgb_image: RGB image in [0, 1] range

    Returns:
        White-balanced image clipped to [0, 1]
    """
    means = rgb_image.mean(axis=(0, 1))
    scale = means[1] / means  # Use green as reference
    balanced = rgb_image * scale[np.newaxis, np.newaxis, :]
    return np.clip(balanced, 0, 1)
