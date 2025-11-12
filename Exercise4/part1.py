import numpy as np
import matplotlib.pyplot as plt
from methods import detect_bayer_pattern

# Load the raw Bayer array
raw = np.load('exercise_4_data/01/IMG_9939.npy')
print('Loaded array of size:', raw.shape)
print()

# Detect Bayer pattern (verbose mode for part 1)
pattern = detect_bayer_pattern(raw, verbose=True)

# Visualize the raw mosaic
plt.figure(figsize=(8, 6))
plt.imshow(raw, cmap='gray')
plt.title(f'Raw Bayer Mosaic – IMG_9939 ({pattern["pattern_name"]} pattern)')
plt.axis('off')
plt.show()
