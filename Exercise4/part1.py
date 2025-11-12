import numpy as np
import matplotlib.pyplot as plt


def detect_bayer_pattern(raw_array):

    # Compute mean value at each 2×2 Bayer offset
    means = {
        (dy, dx): raw_array[dy::2, dx::2].mean()
        for dy in (0, 1) for dx in (0, 1)
    }

    print('Mean values at offsets (dy,dx):')
    for (dy, dx), m in means.items():
        print(f'  Offset ({dy},{dx}): {m:.2f}')

    # Sort by mean (highest to lowest)
    sorted_means = sorted(means.items(), key=lambda x: x[1], reverse=True)

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

    H, W = shape

    r_mask = np.zeros((H, W), bool)
    g_mask = np.zeros((H, W), bool)
    b_mask = np.zeros((H, W), bool)

    # Assign based on detected pattern
    red_offset = pattern['red']
    green1_offset = pattern['green1']
    green2_offset = pattern['green2']
    blue_offset = pattern['blue']

    r_mask[red_offset[0]::2, red_offset[1]::2] = True
    g_mask[green1_offset[0]::2, green1_offset[1]::2] = True
    g_mask[green2_offset[0]::2, green2_offset[1]::2] = True
    b_mask[blue_offset[0]::2, blue_offset[1]::2] = True

    return r_mask, g_mask, b_mask


if __name__ == '__main__':
    # 1. Load the raw Bayer array
    raw = np.load('exercise_4_data/01/IMG_9939.npy')
    print('Loaded array of size:', raw.shape)
    print()

    # 2. Detect Bayer pattern
    pattern = detect_bayer_pattern(raw)

    # 3. Visualize the raw mosaic
    plt.figure(figsize=(8, 6))
    plt.imshow(raw, cmap='gray')
    plt.title(
        f'Raw Bayer Mosaic – IMG_9939 ({pattern["pattern_name"]} pattern)')
    plt.axis('off')
    plt.show()
