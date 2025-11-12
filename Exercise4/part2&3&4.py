import matplotlib.pyplot as plt
import numpy as np
import rawpy
from scipy.ndimage import convolve
from part1 import detect_bayer_pattern, create_bayer_masks

#############  PART 2: DEMOSAICING ##############

with rawpy.imread('exercise_4_data/02/IMG_4782.CR3') as rawp:
    raw_arr = rawp.raw_image_visible.astype(np.float32)

H, W = raw_arr.shape
print(f'Loaded raw image: {H} x {W}\n')

# Detect Bayer pattern
pattern = detect_bayer_pattern(raw_arr)
r_mask, g_mask, b_mask = create_bayer_masks((H, W), pattern)

# Extract channels
R = np.zeros_like(raw_arr)
R[r_mask] = raw_arr[r_mask]
G = np.zeros_like(raw_arr)
G[g_mask] = raw_arr[g_mask]
B = np.zeros_like(raw_arr)
B[b_mask] = raw_arr[b_mask]

# Interpolation
kernel = np.ones((3, 3))


def interp(chan, mask):
    num = convolve(chan, kernel, mode='mirror')
    denom = convolve(mask.astype(np.float32), kernel, mode='mirror')
    return num / np.maximum(denom, 1e-6)


R_i = interp(R, r_mask)
G_i = interp(G, g_mask)
B_i = interp(B, b_mask)

rgb_linear = np.stack([R_i, G_i, B_i], axis=2)

# Visualization
p_low, p_high = np.percentile(rgb_linear, [0.1, 99.9])
rgb_display = np.clip((rgb_linear - p_low) / (p_high - p_low), 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(raw_arr, cmap='gray', vmin=0, vmax=np.percentile(raw_arr, 99))
axes[0].set_title('Raw Bayer Mosaic')
axes[0].axis('off')
axes[1].imshow(rgb_display)
axes[1].set_title('Demosaiced (Linear)')
axes[1].axis('off')
plt.tight_layout()
plt.show()

#############  PART 3: GAMMA CORRECTION ##############


def improve_luminosity(rgb_linear, p_low=0.01, p_high=99.99, gamma=0.3, alpha=10.0):
    norm = np.zeros_like(rgb_linear, dtype=np.float32)

    # Percentile stretch per channel
    for c in range(3):
        lo = np.percentile(rgb_linear[:, :, c], p_low)
        hi = np.percentile(rgb_linear[:, :, c], p_high)
        norm[:, :, c] = np.clip((rgb_linear[:, :, c] - lo) / (hi - lo), 0, 1)

    # Gamma curve
    gamma_corr = np.power(norm, gamma)

    # Log curve
    log_corr = np.log1p(alpha * norm) / np.log1p(alpha)

    return gamma_corr, log_corr


out_gamma, out_log = improve_luminosity(rgb_linear)

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(np.clip(rgb_linear / np.max(rgb_linear), 0, 1))
axs[0].set_title('Linear')
axs[0].axis('off')
axs[1].imshow(out_gamma)
axs[1].set_title('Gamma (γ=0.3)')
axs[1].axis('off')
axs[2].imshow(out_log)
axs[2].set_title('Log (α=10)')
axs[2].axis('off')
plt.tight_layout()
plt.show()

#############  PART 4: WHITE BALANCE ##############


def gray_world_white_balance(img):
    means = img.reshape(-1, 3).mean(axis=0)
    scale = means[1] / means
    wb = img * scale[np.newaxis, np.newaxis, :]
    return np.clip(wb, 0, 1)


wb_img = gray_world_white_balance(out_gamma)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(out_gamma)
axes[0].set_title('After Gamma')
axes[0].axis('off')
axes[1].imshow(wb_img)
axes[1].set_title('After White Balance')
axes[1].axis('off')
plt.tight_layout()
plt.show()
