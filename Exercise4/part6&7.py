import os
import numpy as np
import matplotlib.pyplot as plt
import rawpy
from scipy.ndimage import gaussian_filter
import imageio
from PIL import Image
from methods import (
    detect_bayer_pattern,
    demosaic,
    gray_world_white_balance,
)

results = 'results'
os.makedirs(results, exist_ok=True)


data_root = 'exercise_4_data'
hdr_input_folder = os.path.join(data_root, '06')
simple_hdr_output_path = os.path.join(results, 'HDR_initial.jpg')

#############  PART 6: Initial HDR Implementation ##############


def combination_hdr(raw_folder):
    """
    Combine differently exposed RAW images into one HDR Bayer mosaic
    using the 'simple replacement' HDR method from the lecture.
    """

    raw_file_list = sorted(
        filename for filename in os.listdir(raw_folder)
        if filename.lower().endswith('.cr3')
    )

    num_images = len(raw_file_list)
    exposure_times = np.array(
        [1.0 / (2 ** i) for i in range(num_images)],
        dtype=np.float32
    )

    print(f"Found {num_images} RAW files")
    print("Exposure times:", exposure_times)

    # Load brightest image
    base_path = os.path.join(raw_folder, raw_file_list[0])
    base_time = exposure_times[0]

    print(f"\nLoading base image: {raw_file_list[0]} (t = {base_time})")

    with rawpy.imread(base_path) as raw_reader:
        hdr_mosaic = raw_reader.raw_image_visible.astype(np.float32)
        black = np.mean(raw_reader.black_level_per_channel)
        white = raw_reader.white_level
        hdr_mosaic = (hdr_mosaic - black) / (white - black)
        hdr_mosaic = np.clip(hdr_mosaic, 0, 1) * 65535

    # Replace saturated pixels with shorter exposures
    for i in range(1, num_images):
        filename = raw_file_list[i]
        exposure = exposure_times[i]

        print(f"\nProcessing {filename} (t = {exposure})")

        path = os.path.join(raw_folder, filename)
        with rawpy.imread(path) as raw_reader:
            raw_img = raw_reader.raw_image_visible.astype(np.float32)
            black = np.mean(raw_reader.black_level_per_channel)
            white = raw_reader.white_level
            raw_img = (raw_img - black) / (white - black)
            raw_img = np.clip(raw_img, 0, 1) * 65535

        # Exposure ratio to base image
        exposure_ratio = base_time / exposure
        scaled_img = raw_img * exposure_ratio

        # Saturation threshold
        threshold = 0.8 * np.max(hdr_mosaic)
        saturated_mask = hdr_mosaic > threshold

        print(f"  Saturated pixels: {np.sum(saturated_mask)}")
        print(f"  Replacing saturated values using {filename}")

        hdr_mosaic[saturated_mask] = scaled_img[saturated_mask]

    return hdr_mosaic


def initial_hdr_simple(raw_folder, output_path):
    """
    Full simple HDR pipeline + log tone mapping + gamma + 8-bit output.
    """

    # 1) HDR combine
    hdr_mosaic = combination_hdr(raw_folder)

    # 2) Demosaic
    pattern = detect_bayer_pattern(hdr_mosaic)
    rgb_hdr = demosaic(hdr_mosaic, pattern=pattern)

    # 3) Normalize + white balance
    rgb_hdr = rgb_hdr / np.percentile(rgb_hdr, 99.5)
    rgb_hdr = np.clip(rgb_hdr, 0, 1)
    rgb_hdr = gray_world_white_balance(rgb_hdr)

    # 4) Log tone mapping (per lecture)
    scale = 100.0
    rgb_scaled = rgb_hdr * scale
    rgb_log = np.log1p(rgb_scaled)
    rgb_log_norm = rgb_log / np.max(rgb_log)

    # 5) Gamma + 8-bit
    gamma = 1 / 2.2
    rgb_gamma = np.power(rgb_log_norm, gamma)
    rgb_8bit = np.clip(rgb_gamma * 255, 0, 255).astype(np.uint8)

    # Save
    imageio.imwrite(output_path, rgb_8bit, quality=98)

    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_8bit)
    plt.title('HDR Simple Replacement Method')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


initial_hdr_simple(hdr_input_folder, simple_hdr_output_path)

#############  PART 7: HDR WITH iCAM06 TONE MAPPING ##############

icam_output_path = os.path.join(results, 'HDR_iCAM06.jpg')


def bilateral_filter(image, spatial_sigma=2.0):
    """
    Simplified bilateral filter using Gaussian smoothing (assignment hint).
    """
    filtered = image.copy()
    for _ in range(2):
        filtered = gaussian_filter(filtered, sigma=spatial_sigma / 2)
    return filtered


def icam06_tone_mapping(rgb_hdr, output_dynamic_range=4.0):
    """
    iCAM06 tone mapping on a linear HDR RGB image.
    """

    eps = 1e-10
    rgb_hdr = np.maximum(rgb_hdr, eps)

    # 1. Intensity (luminance)
    intensity = (
        20 * rgb_hdr[:, :, 0] +
        40 * rgb_hdr[:, :, 1] +
        rgb_hdr[:, :, 2]
    ) / 61.0
    intensity = np.maximum(intensity, eps)

    # 2. Chromaticity ratios
    r_ratio = rgb_hdr[:, :, 0] / intensity
    g_ratio = rgb_hdr[:, :, 1] / intensity
    b_ratio = rgb_hdr[:, :, 2] / intensity

    # 3. Log base layer
    log_intensity = np.log(intensity)
    log_base = bilateral_filter(log_intensity, spatial_sigma=5.0)

    # 4. Details
    log_detail = log_intensity - log_base

    # 5. Compression
    bmin = np.min(log_base)
    bmax = np.max(log_base)
    compression = np.log(output_dynamic_range) / (bmax - bmin)
    log_offset = -bmax * compression

    # 6. Output intensity
    log_output = log_base * compression + log_offset + log_detail
    intensity_out = np.exp(log_output)

    # 7. Reconstruct RGB
    rgb_out = np.zeros_like(rgb_hdr)
    rgb_out[:, :, 0] = r_ratio * intensity_out
    rgb_out[:, :, 1] = g_ratio * intensity_out
    rgb_out[:, :, 2] = b_ratio * intensity_out

    return rgb_out


def icam06_hdr(raw_folder, output_path):
    """
    Full HDR merging + iCAM06 tone mapping.
    """

    raw_file_list = sorted(
        filename for filename in os.listdir(raw_folder)
        if filename.lower().endswith('.cr3')
    )

    num_images = len(raw_file_list)
    exposure_times = np.array([1.0 / (2 ** i) for i in range(num_images)],
                              dtype=np.float32)

    print(f"\nFound {num_images} RAW images for iCAM06 HDR")
    print("Exposure times:", exposure_times)

    hdr_sum = None
    weight_sum = None

    # HDR accumulation
    for filename, t in zip(raw_file_list, exposure_times):
        print(f"Processing {filename} (t = {t})")

        full_path = os.path.join(raw_folder, filename)

        with rawpy.imread(full_path) as raw_reader:
            raw_img = raw_reader.raw_image_visible.astype(np.float32)
            black = np.mean(raw_reader.black_level_per_channel)
            white = raw_reader.white_level
            raw_img = (raw_img - black) / (white - black)
            raw_img = np.clip(raw_img, 0, 1) * 65535

        if hdr_sum is None:
            hdr_sum = raw_img / t
            weight_sum = np.full_like(raw_img, 1.0 / t)
        else:
            hdr_sum += raw_img / t
            weight_sum += 1.0 / t

    # Radiance map
    radiance = hdr_sum / weight_sum

    # Demosaic
    pattern = detect_bayer_pattern(radiance)
    rgb_hdr = demosaic(radiance, pattern=pattern)

    # White balance + normalize
    rgb_hdr = rgb_hdr / np.percentile(rgb_hdr, 95)
    rgb_hdr = np.clip(rgb_hdr, 0, 1)
    rgb_hdr = gray_world_white_balance(rgb_hdr)

    # Tone mapping
    print("\nApplying iCAM06 tone mapping...")
    rgb_hdr_scaled = rgb_hdr * 100.0
    rgb_tm = icam06_tone_mapping(rgb_hdr_scaled)

    # Normalize per channel
    rgb_norm = np.zeros_like(rgb_tm)
    for c in range(3):
        ch = rgb_tm[:, :, c]
        lo = np.percentile(ch, 0.5)
        hi = np.percentile(ch, 99.5)
        rgb_norm[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)

    # Gamma + 8-bit
    gamma = 1 / 2.2
    rgb_gamma = np.power(rgb_norm, gamma)
    rgb_8bit = (rgb_gamma * 255).astype(np.uint8)

    # Save
    imageio.imwrite(output_path, rgb_8bit, quality=95)

    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_8bit)
    plt.title('HDR with iCAM06 Tone Mapping')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Run Part 7
icam06_hdr(hdr_input_folder, icam_output_path)
