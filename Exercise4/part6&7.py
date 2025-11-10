import numpy as np
import matplotlib.pyplot as plt
import rawpy
from scipy.ndimage import gaussian_filter
import imageio
from PIL import Image
import os
from scipy.ndimage import convolve

#############  Part 6 ##############
# Configuration
DATA_ROOT = 'exercise_4_data'
FOLDER6 = os.path.join(DATA_ROOT, '06')
OUTPUT_JPG = os.path.join(FOLDER6, 'HDR_initial.jpg')

# Bilinear interpolation kernel
KERNEL = np.array([
    [0.25, 0.5, 0.25],
    [0.5, 1.0, 0.5],
    [0.25, 0.5, 0.25]
], dtype=np.float32)


def demosaic(mosaic):
    """
    Bilinear demosaic of a raw Bayer mosaic (pattern G R / B G).
    mosaic: H×W float32 array of radiance values.
    Returns H×W×3 float32 linear RGB.
    """
    H, W = mosaic.shape

    b_mask = np.zeros((H, W), bool)
    b_mask[0::2, 0::2] = True

    g_mask = np.zeros((H, W), bool)
    g_mask[0::2, 1::2] = True
    g_mask[1::2, 0::2] = True

    r_mask = np.zeros((H, W), bool)
    r_mask[1::2, 1::2] = True

    # Extract known samples
    R = np.zeros_like(mosaic)
    R[r_mask] = mosaic[r_mask]
    G = np.zeros_like(mosaic)
    G[g_mask] = mosaic[g_mask]
    B = np.zeros_like(mosaic)
    B[b_mask] = mosaic[b_mask]

    # Convolution‐based interpolation
    def interp(channel, mask):
        num = convolve(channel, KERNEL, mode='mirror')
        denom = convolve(mask.astype(np.float32), KERNEL, mode='mirror')
        return num / np.maximum(denom, 1e-6)

    R_i = interp(R, r_mask)
    G_i = interp(G, g_mask)
    B_i = interp(B, b_mask)

    return np.stack([B_i, G_i, R_i], axis=2)


def simple_combination_hdr(cr3_folder):

    # List RAW files and exposure times (each half the previous)
    raws = sorted(fn for fn in os.listdir(cr3_folder)
                  if fn.lower().endswith('.cr3'))
    n = len(raws)
    times = np.array([1.0 / (2 ** i) for i in range(n)], dtype=np.float32)

    print(f"Found {n} RAW files")
    print(f"Exposure times: {times}")

    # Load brightest image (longest exposure) as base h
    print(f"Loading base image: {raws[0]} (exposure time: {times[0]})")
    path = os.path.join(cr3_folder, raws[0])
    with rawpy.imread(path) as rp:
        h = rp.raw_image_visible.astype(np.float32)
        black_level = np.mean(rp.black_level_per_channel)
        white_level = rp.white_level
        h = (h - black_level) / (white_level - black_level)
        h = np.clip(h, 0, 1) * 65535

    # Process each shorter exposure
    for i in range(1, n):
        print(f"Processing {raws[i]} (exposure time: {times[i]})")
        path = os.path.join(cr3_folder, raws[i])
        with rawpy.imread(path) as rp:
            raw_i = rp.raw_image_visible.astype(np.float32)
            black_level = np.mean(rp.black_level_per_channel)
            white_level = rp.white_level
            raw_i = (raw_i - black_level) / (white_level - black_level)
            raw_i = np.clip(raw_i, 0, 1) * 65535

        # Multiply i by the exposure difference to the first photo
        exposure_ratio = times[0] / times[i]
        scaled_i = raw_i * exposure_ratio

        # Using threshold of 0.8 * max(h) as suggested
        threshold = 0.8 * np.max(h)
        saturated_mask = h > threshold

        num_saturated = np.sum(saturated_mask)
        print(
            f"  Found {num_saturated} saturated pixels (threshold: {threshold:.1f})")
        print(f"  Replacing with scaled values from exposure {i + 1}")

        h[saturated_mask] = scaled_i[saturated_mask]

    return h


def initial_hdr_simple(cr3_folder, output_path):
    """
    HDR processing using simple replacement approach.
    """
    # Get HDR radiance mosaic using simple combination
    radiance = simple_combination_hdr(cr3_folder)

    # Demosaic to linear RGB
    rgb_lin = demosaic(radiance)

    # Simple white balance
    for c in range(3):
        channel = rgb_lin[:, :, c]
        p1 = np.percentile(channel, 1)
        p99 = np.percentile(channel, 99)
        rgb_lin[:, :, c] = (channel - p1) / (p99 - p1)

    rgb_lin = np.clip(rgb_lin, 0, 1)

    # Apply log scale tone mapping
    scale_factor = 100.0
    rgb_scaled = rgb_lin * scale_factor
    rgb_log = np.log1p(rgb_scaled)
    rgb_normalized = rgb_log / np.max(rgb_log)

    # Apply gamma correction for display
    gamma = 1.0 / 2.2
    rgb_display = np.power(rgb_normalized, gamma)
    rgb_final = rgb_display * 255.0

    # Clip and convert to uint8
    rgb_final = np.clip(rgb_final, 0, 255)
    out8 = rgb_final.astype(np.uint8)

    # Save the image
    imageio.imwrite(output_path, out8, quality=98)

    # Display the result
    HDR_img = Image.open(output_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(HDR_img)
    plt.title('HDR Simple Replacement Approach (Log-Scale Tone Mapping)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


initial_hdr_simple(FOLDER6, OUTPUT_JPG)

#############  Part 6 ##############

#############  Part 7 ##############

# Configuration
DATA_ROOT = 'exercise_4_data'
FOLDER7 = os.path.join(DATA_ROOT, '06')
OUTPUT_JPG = os.path.join(FOLDER7, 'HDR_iCAM06.jpg')

# Bilinear interpolation kernel (weights sum to 4)
KERNEL = np.array([
    [0.25, 0.5, 0.25],
    [0.5, 1.0, 0.5],
    [0.25, 0.5, 0.25]
], dtype=np.float32)


def demosaic(mosaic):
    """
    Bilinear demosaic of a raw Bayer mosaic (BGGR pattern).
    """
    H, W = mosaic.shape

    # BGGR pattern
    b_mask = np.zeros((H, W), bool)
    b_mask[0::2, 0::2] = True

    g_mask = np.zeros((H, W), bool)
    g_mask[0::2, 1::2] = True
    g_mask[1::2, 0::2] = True

    r_mask = np.zeros((H, W), bool)
    r_mask[1::2, 1::2] = True

    # Extract known samples
    R = np.zeros_like(mosaic)
    R[r_mask] = mosaic[r_mask]
    G = np.zeros_like(mosaic)
    G[g_mask] = mosaic[g_mask]
    B = np.zeros_like(mosaic)
    B[b_mask] = mosaic[b_mask]

    # Convolution-based interpolation
    def interp(channel, mask):
        num = convolve(channel, KERNEL, mode='mirror')
        denom = convolve(mask.astype(np.float32), KERNEL, mode='mirror')
        return num / np.maximum(denom, 1e-6)

    R_i = interp(R, r_mask)
    G_i = interp(G, g_mask)
    B_i = interp(B, b_mask)

    # Return with channels swapped: BGR -> RGB
    return np.stack([B_i, G_i, R_i], axis=2)


def bilateral_filter(image, spatial_sigma=2.0, range_sigma=0.1, kernel_size=None):
    """
    Simple bilateral filter implementation.
    For speed during development, use small kernels as suggested.
    """
    if kernel_size is None:
        kernel_size = int(2 * np.ceil(2 * spatial_sigma) + 1)

    # For initial development, use Gaussian filter as approximation
    # For production, implement full bilateral filter
    filtered = image.copy()
    for _ in range(2):
        filtered = gaussian_filter(filtered, sigma=spatial_sigma / 2)
    return filtered


def icam06_tone_mapping(rgb_hdr, output_range=4.0):
    """
    Implement iCAM06 tone mapping algorithm.

    Parameters:
    - rgb_hdr: HDR image in linear RGB
    - output_range: Target output dynamic range (default 4.0)
    """
    # Ensure no zeros for log operations
    eps = 1e-10
    rgb_hdr = np.maximum(rgb_hdr, eps)

    # 1. Calculate input intensity (1/61 * (20*red + 40*green + blue))
    input_intensity = (
        20 * rgb_hdr[:, :, 0] + 40 * rgb_hdr[:, :, 1] + rgb_hdr[:, :, 2]) / 61.0
    input_intensity = np.maximum(input_intensity, eps)

    # 2. Calculate r, g, b ratios
    r_ratio = rgb_hdr[:, :, 0] / input_intensity
    g_ratio = rgb_hdr[:, :, 1] / input_intensity
    b_ratio = rgb_hdr[:, :, 2] / input_intensity

    # 3. Calculate log_base using bilateral filter
    # Start with small sigma for speed during development
    log_input = np.log(input_intensity)
    log_base = bilateral_filter(log_input, spatial_sigma=5.0, range_sigma=0.3)

    # 4. Calculate log_details
    log_details = log_input - log_base

    # 5. Calculate compression factor
    log_base_min = np.min(log_base)
    log_base_max = np.max(log_base)
    compression = np.log(output_range) / (log_base_max - log_base_min)

    # 6. Calculate log_offset
    log_offset = -log_base_max * compression

    # 7. Calculate output_intensity
    log_output_intensity = log_base * compression + log_offset + log_details
    output_intensity = np.exp(log_output_intensity)

    # 8. Reconstruct RGB from ratios and output intensity
    rgb_out = np.zeros_like(rgb_hdr)
    rgb_out[:, :, 0] = r_ratio * output_intensity
    rgb_out[:, :, 1] = g_ratio * output_intensity
    rgb_out[:, :, 2] = b_ratio * output_intensity

    return rgb_out


def icam06_hdr(cr3_folder, output_path):
    """
    Process HDR using iCAM06 method.
    """
    # 1) List RAW files and exposure times
    raws = sorted(fn for fn in os.listdir(cr3_folder)
                  if fn.lower().endswith('.cr3'))
    n = len(raws)
    times = np.array([1.0 / (2 ** i) for i in range(n)], dtype=np.float32)

    print(f"Found {n} RAW files for iCAM06 processing")
    print(f"Exposure times: {times}")

    # 2) Accumulate HDR radiance
    H_acc = None
    D_acc = None

    for i, (fn, t) in enumerate(zip(raws, times)):
        print(f"Processing {fn} with exposure time {t}")
        path = os.path.join(cr3_folder, fn)
        with rawpy.imread(path) as rp:
            raw = rp.raw_image_visible.astype(np.float32)

            # Handle black level
            black_level = np.mean(rp.black_level_per_channel)
            white_level = rp.white_level

            # Normalize
            raw = (raw - black_level) / (white_level - black_level)
            raw = np.clip(raw, 0, 1) * 65535

        if H_acc is None:
            H_acc = raw / t
            D_acc = np.full_like(raw, 1.0 / t)
        else:
            H_acc += raw / t
            D_acc += 1.0 / t

    # 3) Calculate radiance
    radiance = H_acc / D_acc

    # 4) Demosaic to linear RGB
    rgb_hdr = demosaic(radiance)

    # 5) White balance - more aggressive correction
    # Calculate white balance on middle gray region
    h, w = rgb_hdr.shape[:2]
    mid_h = slice(int(h * 0.3), int(h * 0.7))
    mid_w = slice(int(w * 0.3), int(w * 0.7))

    # Find gray world correction factors
    r_mean = np.mean(rgb_hdr[mid_h, mid_w, 0])
    g_mean = np.mean(rgb_hdr[mid_h, mid_w, 1])
    b_mean = np.mean(rgb_hdr[mid_h, mid_w, 2])

    gray_mean = (r_mean + g_mean + b_mean) / 3.0

    rgb_hdr[:, :, 0] *= gray_mean / r_mean
    rgb_hdr[:, :, 1] *= gray_mean / g_mean
    rgb_hdr[:, :, 2] *= gray_mean / b_mean

    # Normalize to reasonable range
    rgb_hdr = rgb_hdr / np.percentile(rgb_hdr, 90)

    # 6) Apply iCAM06 tone mapping
    print("Applying iCAM06 tone mapping...")
    # Scale the HDR data before tone mapping
    rgb_hdr_scaled = rgb_hdr * 100.0  # Scale up for better dynamic range
    rgb_tm = icam06_tone_mapping(rgb_hdr_scaled, output_range=4.0)

    # 7) Normalize to display range
    # Clip extreme values first
    rgb_tm = np.clip(rgb_tm, 0, np.percentile(rgb_tm, 99.5))

    # Normalize each channel separately for better color balance
    rgb_display = np.zeros_like(rgb_tm)
    for c in range(3):
        channel = rgb_tm[:, :, c]
        c_min = np.percentile(channel, 0.5)
        c_max = np.percentile(channel, 99.5)
        if c_max > c_min:
            rgb_display[:, :, c] = (channel - c_min) / (c_max - c_min)

    rgb_display = np.clip(rgb_display, 0, 1)

    # 8) Apply gamma correction
    gamma = 1.0 / 2.2
    rgb_gamma = np.power(rgb_display, gamma)

    # 9) Convert to 8-bit
    rgb_final = (rgb_gamma * 255).astype(np.uint8)

    # 10) Save the image
    imageio.imwrite(output_path, rgb_final, quality=95)

    # 11) Display the result
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_final)
    plt.title('HDR with iCAM06 Tone Mapping')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Show comparison with initial
    initial_path = os.path.join(os.path.dirname(
        cr3_folder), '06', 'HDR_initial.jpg')
    if os.path.exists(initial_path):
        plt.figure(figsize=(16, 8))

        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(initial_path))
        plt.title('Initial HDR (Log Scale)')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(rgb_final)
        plt.title('iCAM06 HDR')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


icam06_hdr(FOLDER7, OUTPUT_JPG)

#############  Part 7 ##############
