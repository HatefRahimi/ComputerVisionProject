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

# 3. Identify likely green channel (highest mean)
green_offset = max(means, key=means.get)
print('Likely GREEN pixels at offset (dy,dx):', green_offset)

# 4. Visualize the raw mosaic
plt.figure(figsize=(6, 6))
plt.imshow(raw, cmap='gray')
plt.title('Raw Bayer Mosaic – IMG_9939')
plt.axis('off')
plt.show()

#############  Part 1 ##############