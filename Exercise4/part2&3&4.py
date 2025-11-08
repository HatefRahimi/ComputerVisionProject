import numpy as np
import rawpy
import imageio
from PIL import Image
from scipy.ndimage import convolve

#############  Part 2 ##############

# 1) Load RAW sensor data as float32
with rawpy.imread('exercise_4_data/02/IMG_4782.CR3') as rawp:
    raw_arr = rawp.raw_image_visible.astype(np.float32)
H, W = raw_arr.shape

# 2) Build Bayer masks (RG/GB pattern)
r_mask = np.zeros((H, W), bool)
g_mask = np.zeros((H, W), bool)
b_mask = np.zeros((H, W), bool)

# R at (0,0)
r_mask[0::2, 0::2] = True
# G at (0,1) and (1,0)
g_mask[0::2, 1::2] = True
g_mask[1::2, 0::2] = True
# B at (1,1)
b_mask[1::2, 1::2] = True

# 3) Allocate channels and assign known samples
R = np.zeros_like(raw_arr)
R[r_mask] = raw_arr[r_mask]
G = np.zeros_like(raw_arr)
G[g_mask] = raw_arr[g_mask]
B = np.zeros_like(raw_arr)
B[b_mask] = raw_arr[b_mask]

# 4) Define bilinear kernel & interpolation helper
kernel = np.array([[0.25, 0.5, 0.25],
                   [0.5, 1.0, 0.5],
                   [0.25, 0.5, 0.25]])


def interp(chan, mask):
    num = convolve(chan, kernel, mode='mirror')
    denom = convolve(mask.astype(np.float32), kernel, mode='mirror')
    return num / np.maximum(denom, 1e-6)


# 5) Interpolate missing values in each channel
R_i = interp(R, r_mask)
G_i = interp(G, g_mask)
B_i = interp(B, b_mask)

# 6) Stack into a single H×W×3 RGB image (float32, still linear)
rgb_linear = np.stack([R_i, G_i, B_i], axis=2)

# Apply Gray-World white balance **before** any gamma or clipping
means = rgb_linear.reshape(-1, 3).mean(axis=0)       # [mean_R, mean_G, mean_B]
scale = means[1] / means                             # normalize relative to green
rgb_linear_wb = rgb_linear * scale[np.newaxis, np.newaxis, :]

# inspect
print('Demosaiced RGB shape:', rgb_linear.shape, 'dtype:', rgb_linear.dtype)


#############  Part 2 ##############

#############  Part 3 ##############

def improve_luminosity(rgb_linear, p_low=0.01, p_high=99.99, gamma=0.3, alpha=10.0):
    """
    - rgb_linear: H×W×3 float32 array from demosaic (linear domain)
    - p_low, p_high: percentiles for normalization
    - gamma: exponent for gamma correction
    - alpha: parameter for alternative log curve

    Returns two H×W×3 arrays (float32), stretched back to the original raw scale:
      • out_gamma: using y = x^gamma
      • out_log:   using y = log(1 + α x) / log(1 + α)
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
    gamma_corr = norm ** gamma
    out_gamma = (gamma_corr * 255.0).astype(np.uint8)

    # 2b) Log curve
    log_corr = np.log1p(alpha * norm) / np.log1p(alpha)
    out_log = (log_corr * 255.0).astype(np.uint8)

    return out_gamma, out_log


out_gamma, out_log = improve_luminosity(rgb_linear_wb)

# Inspect percentile stretch values:
print(f'Stretch bounds → lo = {np.percentile(rgb_linear, 0.01):.2f}, '
      f'hi = {np.percentile(rgb_linear, 99.99):.2f}')

# visualize the results with matplotlib:
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(np.clip(rgb_linear / np.max(rgb_linear), 0, 1))
axs[0].set_title('Linear Input (scaled)')
axs[1].imshow(np.clip((out_gamma - out_gamma.min()) / (out_gamma.max() - out_gamma.min()), 0, 1))
axs[1].set_title('Gamma γ=0.3')
axs[2].imshow(np.clip((out_log - out_log.min()) / (out_log.max() - out_log.min()), 0, 1))
axs[2].set_title('Logarithmic α=10')
for ax in axs:
    ax.axis('off')
plt.show()


#############  Part 3 ##############

#############  Part 4 ##############

def gray_world_white_balance(img):
    """
    img: H×W×3 float32 array, expected in the [0,255] scale (or similar).
    Returns an H×W×3 uint8 image after gray-world balance and clipping.
    """
    # 1) Compute per-channel means
    means = img.reshape(-1, 3).mean(axis=0)  # [mean_R, mean_G, mean_B]

    # 2) Scale factors to make green mean = 1
    #    (so all channels are scaled relative to green)
    scale = means[1] / means  # [s_R, s_G=1, s_B]

    # 3) Apply and clip
    wb = img * scale[np.newaxis, np.newaxis, :]
    wb_clipped = np.clip(wb, 0, 255).astype(np.uint8)

    return wb_clipped


# scaled back into the original raw value range. First normalize it to [0,255]:
out_gamma_8bit = np.clip((out_gamma - out_gamma.min()) /
                         (out_gamma.max() - out_gamma.min()) * 255, 0, 255).astype(np.float32)

# Now white-balance:
wb_img = gray_world_white_balance(out_gamma_8bit)

# Save and display:
imageio.imwrite('exercise_4_data/02/IMG_4782_wb.jpg', wb_img, quality=98)
wb_img = Image.open('exercise_4_data/02/IMG_4782_wb.jpg')

# 2) Display
plt.figure(figsize=(8, 6))
plt.imshow(wb_img)
plt.axis('off')
plt.title('IMG_4782 – After Gray-World White Balance')
plt.show()

#############  Part 4 ##############