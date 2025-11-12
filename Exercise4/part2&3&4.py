import matplotlib.pyplot as plt
import numpy as np
import rawpy
import imageio
from scipy.ndimage import convolve
from part1 import detect_bayer_pattern, create_bayer_masks

#############  EXERCISE 2: LOAD RAW DATA ##############

with rawpy.imread('exercise_4_data/02/IMG_4782.CR3') as rawp:
    raw_arr = rawp.raw_image_visible.astype(np.float32)

H, W = raw_arr.shape
print(f'Loaded raw image: {H} x {W}')
print()

#############  DETECT BAYER PATTERN ##############

pattern = detect_bayer_pattern(raw_arr)
r_mask, g_mask, b_mask = create_bayer_masks((H, W), pattern)

#############  EXTRACT CHANNELS ##############

R = np.zeros_like(raw_arr)
R[r_mask] = raw_arr[r_mask]
G = np.zeros_like(raw_arr)
G[g_mask] = raw_arr[g_mask]
B = np.zeros_like(raw_arr)
B[b_mask] = raw_arr[b_mask]

#############  INTERPOLATION ##############

# Simplest possible kernel - uniform 3x3 average
kernel = np.ones((3, 3))


def interp(chan, mask):
    """Simple averaging interpolation for Bayer demosaicing."""
    num = convolve(chan, kernel, mode='mirror')
    denom = convolve(mask.astype(np.float32), kernel, mode='mirror')
    return num / np.maximum(denom, 1e-6)


R_i = interp(R, r_mask)
G_i = interp(G, g_mask)
B_i = interp(B, b_mask)

rgb_linear = np.stack([R_i, G_i, B_i], axis=2)

#############  EXERCISE 2: VISUALIZATION ##############

# Create display version with percentile clipping
p_low, p_high = np.percentile(rgb_linear, [0.1, 99.9])
rgb_display = np.clip((rgb_linear - p_low) / (p_high - p_low), 0, 1)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Raw Bayer data (grayscale)
axes[0].imshow(raw_arr, cmap='gray', vmin=0, vmax=np.percentile(raw_arr, 99))
axes[0].set_title('Raw Bayer Mosaic Data')
axes[0].axis('off')

# 2. Demosaiced (linear, normalized for display)
axes[1].imshow(rgb_display)
axes[1].set_title('Demosaiced Image (Linear, No Gamma/White Balance)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('exercise2_demosaiced_result.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'\nDemosaiced image shape: {rgb_linear.shape}')
print(f'Value range: [{rgb_linear.min():.2f}, {rgb_linear.max():.2f}]')
print(f'Mean per channel: R={rgb_linear[:, :, 0].mean():.2f}, '
      f'G={rgb_linear[:, :, 1].mean():.2f}, B={rgb_linear[:, :, 2].mean():.2f}')

# Save for later exercises
np.save('demosaiced_linear.npy', rgb_linear)
print('\nSaved linear RGB data to: demosaiced_linear.npy')

print('\n' + '='*60)
print('EXERCISE 2 COMPLETE')
print('='*60 + '\n')

#############  EXERCISE 3: IMPROVE LUMINOSITY ##############


def improve_luminosity(rgb_linear, p_low=0.01, p_high=99.99, gamma=0.3, alpha=10.0):
    """
    Apply gamma correction and alternative curve to improve luminosity.

    Parameters:
    -----------
    rgb_linear : np.ndarray
        H×W×3 float32 array from demosaic (linear domain)
    p_low, p_high : float
        Percentiles for normalization
    gamma : float
        Exponent for gamma correction
    alpha : float
        Parameter for alternative log curve

    Returns:
    --------
    tuple : (out_gamma, out_log) - H×W×3 float32 arrays in [0,1] range
    """
    H, W, _ = rgb_linear.shape
    norm = np.zeros_like(rgb_linear, dtype=np.float32)

    # 1) Per-channel percentile stretch into [0,1]
    for c in range(3):
        lo = np.percentile(rgb_linear[:, :, c], p_low)
        hi = np.percentile(rgb_linear[:, :, c], p_high)
        chan = (rgb_linear[:, :, c] - lo) / (hi - lo)
        norm[:, :, c] = np.clip(chan, 0.0, 1.0)

    # 2a) Gamma curve
    gamma_corr = np.power(norm, gamma)

    # 2b) Log curve
    log_corr = np.log1p(alpha * norm) / np.log1p(alpha)

    return gamma_corr, log_corr


# Apply luminosity correction
out_gamma, out_log = improve_luminosity(rgb_linear, gamma=0.3, alpha=10.0)

# Inspect percentile stretch values
print('\nEXERCISE 3: Luminosity Correction')
print(
    f'Original value range: [{rgb_linear.min():.2f}, {rgb_linear.max():.2f}]')
print(f'After gamma (γ=0.3): [{out_gamma.min():.3f}, {out_gamma.max():.3f}]')
print(f'After log (α=10): [{out_log.min():.3f}, {out_log.max():.3f}]')

# Visualize the results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Original (normalized for display)
axs[0].imshow(np.clip(rgb_linear / np.max(rgb_linear), 0, 1))
axs[0].set_title('Linear Input (normalized)')
axs[0].axis('off')

# Gamma corrected
axs[1].imshow(np.clip(out_gamma, 0, 1))
axs[1].set_title('Gamma Correction (γ=0.3)')
axs[1].axis('off')

# Log corrected
axs[2].imshow(np.clip(out_log, 0, 1))
axs[2].set_title('Logarithmic (α=10)')
axs[2].axis('off')

plt.tight_layout()
plt.savefig('exercise3_luminosity_correction.png',
            dpi=150, bbox_inches='tight')
plt.show()

print('\n' + '='*60)
print('EXERCISE 3 COMPLETE')
print('='*60 + '\n')

#############  EXERCISE 4: WHITE BALANCE ##############


def gray_world_white_balance(img):
    """
    Apply gray-world white balance.

    Parameters:
    -----------
    img : np.ndarray
        H×W×3 float32 array in [0,1] range

    Returns:
    --------
    np.ndarray : H×W×3 float32 array after white balance, clipped to [0,1]
    """
    # 1) Compute per-channel means
    means = img.reshape(-1, 3).mean(axis=0)  # [mean_R, mean_G, mean_B]

    # 2) Scale factors relative to green
    scale = means[1] / means  # [s_R, s_G=1, s_B]

    print(
        f'  Channel means: R={means[0]:.4f}, G={means[1]:.4f}, B={means[2]:.4f}')
    print(
        f'  Scale factors: R={scale[0]:.4f}, G={scale[1]:.4f}, B={scale[2]:.4f}')

    # 3) Apply and clip
    wb = img * scale[np.newaxis, np.newaxis, :]
    wb_clipped = np.clip(wb, 0, 1)

    return wb_clipped


# Apply white balance to gamma-corrected image
print('EXERCISE 4: White Balance')
wb_img = gray_world_white_balance(out_gamma)

# Convert to uint8 for saving
wb_img_uint8 = (wb_img * 255).astype(np.uint8)

# Save
imageio.imwrite('exercise4_white_balanced.jpg', wb_img_uint8, quality=98)
print('\nSaved white-balanced image to: exercise4_white_balanced.jpg')

# Display comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Before white balance (with gamma)
axes[0].imshow(out_gamma)
axes[0].set_title('After Gamma Correction\n(No White Balance)')
axes[0].axis('off')

# After white balance
axes[1].imshow(wb_img)
axes[1].set_title('After Gray-World White Balance')
axes[1].axis('off')

# Side by side zoom (center crop)
h_crop = H // 4
w_crop = W // 4
crop = (slice(h_crop, 3*h_crop), slice(w_crop, 3*w_crop))
axes[2].imshow(wb_img[crop])
axes[2].set_title('White Balanced (Center Crop)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('exercise4_white_balance_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()

print('\n' + '='*60)
print('EXERCISE 4 COMPLETE')
print('='*60 + '\n')

#############  FINAL PIPELINE SUMMARY ##############

print('\n' + '='*60)
print('COMPLETE PIPELINE SUMMARY')
print('='*60)
print(f'Input:  Raw Bayer mosaic ({H}×{W})')
print(f'Step 1: Demosaicing → RGB linear ({H}×{W}×3)')
print(f'Step 2: Gamma correction (γ=0.3)')
print(f'Step 3: Gray-world white balance')
print(f'Output: Final RGB image saved as exercise4_white_balanced.jpg')
print('='*60)
