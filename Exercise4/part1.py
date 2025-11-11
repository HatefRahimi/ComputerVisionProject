import numpy as np
import matplotlib.pyplot as plt

#############  Part 1 ##############

# 1. Load the raw Bayer array
raw = np.load('exercise_4_data/01/IMG_9939.npy')
print('Loaded array of size:', raw.shape)

# 2. Compute mean value at each 2×2 Bayer offset
means = {
    (dy, dx): raw[dy::2, dx::2].mean()
    for dy in (0, 1) for dx in (0, 1)
}
print('Mean values at offsets (dy,dx):')
for (dy, dx), m in means.items():
    print(f'  Offset ({dy},{dx}): {m:.2f}')


sorted_means = sorted(means.items(), key=lambda x: x[1], reverse=True)

print('\nSorted by mean (highest to lowest):')
for (dy, dx), m in sorted_means:
    print(f'  Offset ({dy},{dx}): {m:.2f}')

green1_offset = sorted_means[0][0]
green2_offset = sorted_means[1][0]
red_offset = sorted_means[2][0]
blue_offset = sorted_means[3][0]

print('\n--- Identified Bayer Pattern ---')
print(f'Green 1: offset {green1_offset}')
print(f'Green 2: offset {green2_offset}')
print(f'Red:     offset {red_offset}')
print(f'Blue:    offset {blue_offset}')

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
print(f'Raw Bayer Mosaic – IMG_9939 ({pattern_name} pattern)')

# 4. Visualize the raw mosaic
plt.figure(figsize=(8, 6))
plt.imshow(raw, cmap='gray')
plt.title('Raw Bayer Mosaic – IMG_9939')
plt.axis('off')
plt.show()

#############  Part 1 ##############
