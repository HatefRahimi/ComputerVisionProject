import numpy as np
from scipy.ndimage import convolve
import numpy as np


def detect_bayer_pattern_fixed(pattern_name, verbose=False):

    # Defined INSIDE the method
    Bayer_pattern = {
        'RGGB': {'red': (0, 0), 'green1': (0, 1), 'green2': (1, 0), 'blue': (1, 1)},
        'BGGR': {'red': (1, 1), 'green1': (0, 1), 'green2': (1, 0), 'blue': (0, 0)},
        'GRBG': {'red': (0, 1), 'green1': (0, 0), 'green2': (1, 1), 'blue': (1, 0)},
        'GBRG': {'red': (1, 0), 'green1': (0, 0), 'green2': (1, 1), 'blue': (0, 1)},
    }

    if pattern_name not in Bayer_pattern:
        raise ValueError(f"Invalid Bayer pattern: {pattern_name}")

    if verbose:
        print(f"Using fixed Bayer pattern: {pattern_name}")

    base = Bayer_pattern[pattern_name]

    return {
        'red': base['red'],
        'green1': base['green1'],
        'green2': base['green2'],
        'blue': base['blue'],
        'pattern_name': pattern_name
    }


def detect_bayer_pattern(raw_array, verbose=False):
    # Compute mean value at each 2×2 Bayer offset
    means = {}
    for dy in (0, 1):
        for dx in (0, 1):
            means[(dy, dx)] = raw_array[dy::2, dx::2].mean()

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

    # Two highest: greens
    green1_offset = sorted_means[0][0]
    green2_offset = sorted_means[1][0]

    # Remaining two are red and blue
    remaining = [sorted_means[2][0], sorted_means[3][0]]

    # Common diagonal patterns
    if (0, 0) in remaining and (1, 1) in remaining:
        # Diagonal pattern: RGGB
        red_offset = (0, 0)
        blue_offset = (1, 1)
    elif (0, 1) in remaining and (1, 0) in remaining:
        # Diagonal pattern: GRBG
        red_offset = (0, 1)
        blue_offset = (1, 0)
    else:
        # Fallback: use brightness order (unreliable)
        red_offset = sorted_means[2][0]
        blue_offset = sorted_means[3][0]
        if verbose:
            print("\nUnusual Bayer pattern detected")
            print("Use the fixed method.")

    # Build pattern string like 'RGGB'
    pattern_map = {
        (0, 0): None,
        (0, 1): None,
        (1, 0): None,
        (1, 1): None
    }

    pattern_map[green1_offset] = 'G'
    pattern_map[green2_offset] = 'G'
    pattern_map[red_offset] = 'R'
    pattern_map[blue_offset] = 'B'

    pattern_name = (
        pattern_map[(0, 0)] +
        pattern_map[(0, 1)] +
        pattern_map[(1, 0)] +
        pattern_map[(1, 1)]
    )

    if verbose:
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
        'pattern_name': pattern_name,
    }


def create_bayer_masks(shape, pattern):
    """
    binary masks for each color channel based on Bayer pattern.
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
    Interpolation formula.
    """
    if kernel is None:
        kernel = np.ones((3, 3))

    num = convolve(channel, kernel, mode='mirror')
    denom = convolve(mask.astype(np.float32), kernel, mode='mirror')
    return num / np.maximum(denom, 1e-6)


def demosaic(raw_data, pattern=None):
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


def gamma_correction(rgb_image, p_low=0.01, p_high=99.99, gamma=0.3):
    """
    gamma correction with percentile-based normalization.
    """
    low_percentile = np.percentile(rgb_image, p_low)
    high_percentile = np.percentile(rgb_image, p_high)

    normalized = (rgb_image - low_percentile) / \
        (high_percentile - low_percentile)
    normalized = np.clip(normalized, 0.0, 1.0)

    return np.power(normalized, gamma)


def gray_world_white_balance(rgb_image):
    """
     gray world white balance assumption.
    """
    means = rgb_image.mean(axis=(0, 1))
    scale = means[1] / means  # Use green as reference
    balanced = rgb_image * scale[np.newaxis, np.newaxis, :]
    return np.clip(balanced, 0, 1)
