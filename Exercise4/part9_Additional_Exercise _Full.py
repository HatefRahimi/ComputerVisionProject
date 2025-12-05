"""
Exercise 9: HDR from JPG Images
Additional Exercise - Estimating Camera Response Function

This implementation recovers HDR images from JPG files by:
1. Estimating the camera's response curve y = f(x)
2. Inverting it to recover linear light values
3. Combining images into HDR
4. Tone mapping for display
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


# ============================================================================
# STEP 1: ESTIMATE CAMERA RESPONSE FUNCTION
# ============================================================================

def estimate_camera_response_curve(image_paths, exposure_times, num_samples=1000):
    """
    Estimate the camera response function from multiple exposures.

    The camera applies: JPG_value = f(linear_light * exposure_time)
    We want to find f and invert it.

    Args:
        image_paths: List of paths to JPG images (ordered from dark to bright)
        exposure_times: Array of relative exposure times
        num_samples: Number of pixel samples to use for estimation

    Returns:
        response_curve: Lookup table mapping JPG values [0-255] to linear values
    """

    print("Loading images for response curve estimation...")
    images = []
    for path in image_paths:
        img = np.array(Image.open(path)).astype(np.float32) / 255.0
        # Convert to grayscale for simplicity
        if len(img.shape) == 3:
            img = 0.299 * img[:, :, 0] + 0.587 * \
                img[:, :, 1] + 0.114 * img[:, :, 2]
        images.append(img)

    images = np.array(images)
    num_images, height, width = images.shape

    print(f"Loaded {num_images} images of size {height}x{width}")

    # Sample random pixels (avoid very dark and very bright)
    np.random.seed(42)
    sample_y = np.random.randint(height//4, 3*height//4, num_samples)
    sample_x = np.random.randint(width//4, 3*width//4, num_samples)

    # Collect pixel values across all exposures
    # Shape: (num_images, num_samples)
    pixel_values = images[:, sample_y, sample_x]

    # For each intensity level, collect observations
    # We'll build a lookup table
    intensity_bins = 256
    response_curve = np.zeros(intensity_bins)
    counts = np.zeros(intensity_bins)

    print("Building response curve...")

    # For each sample pixel
    for s in range(num_samples):
        # Get pixel values across all exposures
        pixel_across_exposures = pixel_values[:, s]

        # Find exposures where pixel is in valid range (not too dark or bright)
        valid_mask = (pixel_across_exposures > 0.05) & (
            pixel_across_exposures < 0.95)

        if np.sum(valid_mask) < 2:
            continue

        # Use median exposure as reference
        valid_exposures = exposure_times[valid_mask]
        valid_pixels = pixel_across_exposures[valid_mask]

        # Estimate linear irradiance (assuming middle exposure is reference)
        median_idx = len(valid_pixels) // 2
        reference_pixel = valid_pixels[median_idx]
        reference_exposure = valid_exposures[median_idx]

        # Estimate linear value
        linear_estimate = reference_pixel / reference_exposure

        # Add to histogram
        for pix_val, exp_time in zip(valid_pixels, valid_exposures):
            bin_idx = int(pix_val * 255)
            if 0 <= bin_idx < intensity_bins:
                response_curve[bin_idx] += linear_estimate * exp_time
                counts[bin_idx] += 1

    # Average the accumulated values
    valid_bins = counts > 0
    response_curve[valid_bins] /= counts[valid_bins]

    # Smooth the response curve
    response_curve = gaussian_filter(response_curve, sigma=2.0)

    # Normalize so that curve(255) = 1
    if response_curve[-1] > 0:
        response_curve = response_curve / response_curve[-1]

    # Ensure monotonicity
    for i in range(1, len(response_curve)):
        if response_curve[i] < response_curve[i-1]:
            response_curve[i] = response_curve[i-1]

    return response_curve


def plot_response_curve(response_curve, output_path='results/response_curve.png'):
    """Plot the estimated camera response function."""

    os.makedirs('results', exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot estimated curve
    x = np.arange(256) / 255.0
    plt.plot(x, response_curve, 'b-', linewidth=2, label='Estimated Response')

    # Plot reference gamma curves
    gamma_22 = x ** 2.2
    gamma_24 = x ** 2.4
    plt.plot(x, gamma_22, 'r--', alpha=0.5, label='γ=2.2 (sRGB)')
    plt.plot(x, gamma_24, 'g--', alpha=0.5, label='γ=2.4')

    plt.xlabel('JPG Pixel Value (normalized)', fontsize=12)
    plt.ylabel('Linear Light Value', fontsize=12)
    plt.title('Camera Response Function', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"Response curve saved to {output_path}")


# ============================================================================
# STEP 2: LINEARIZE JPG IMAGES
# ============================================================================

def linearize_image(jpg_image, response_curve):
    """
    Convert JPG image to linear light values using response curve.

    Args:
        jpg_image: Image array in [0, 1] range
        response_curve: Lookup table from estimate_camera_response_curve

    Returns:
        Linear image
    """

    # Quantize to 256 levels for lookup
    jpg_quantized = (jpg_image * 255).astype(np.int32)
    jpg_quantized = np.clip(jpg_quantized, 0, 255)

    # Apply inverse response function
    linear_image = response_curve[jpg_quantized]

    return linear_image


# ============================================================================
# STEP 3: COMBINE INTO HDR
# ============================================================================

def merge_exposures_jpg(image_paths, exposure_times, response_curve):
    """
    Merge multiple JPG exposures into HDR image.

    Args:
        image_paths: List of JPG file paths
        exposure_times: Array of relative exposure times
        response_curve: Camera response function

    Returns:
        HDR RGB image
    """

    print(f"\nMerging {len(image_paths)} exposures into HDR...")

    # Load first image to get dimensions
    first_img = np.array(Image.open(image_paths[0])).astype(np.float32) / 255.0
    height, width = first_img.shape[:2]

    # Initialize HDR accumulator
    hdr_image = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width, 3), dtype=np.float32)

    # Weight function: prefer mid-range values
    def weight_function(pixel_value):
        """Tent function: weight is highest at 0.5, decreases to 0 at edges."""
        return 1.0 - np.abs(2.0 * pixel_value - 1.0) ** 2

    for i, (path, exp_time) in enumerate(zip(image_paths, exposure_times)):
        print(
            f"  Processing {os.path.basename(path)} (exposure={exp_time:.4f})...")

        # Load and linearize
        jpg = np.array(Image.open(path)).astype(np.float32) / 255.0

        if len(jpg.shape) == 2:
            jpg = np.stack([jpg, jpg, jpg], axis=2)

        # Linearize each channel
        linear = np.zeros_like(jpg)
        for c in range(3):
            linear[:, :, c] = linearize_image(jpg[:, :, c], response_curve)

        # Compute weights based on original JPG values
        weights = weight_function(jpg)

        # Accumulate weighted irradiance
        irradiance = linear / exp_time
        hdr_image += weights * irradiance
        weight_sum += weights

    # Normalize by total weight
    valid_mask = weight_sum > 1e-6
    hdr_image[valid_mask] /= weight_sum[valid_mask]

    print("HDR merge complete!")

    return hdr_image


# ============================================================================
# STEP 4: TONE MAPPING
# ============================================================================

def tone_map_reinhard(hdr_image, key_value=0.18, white_point=None):
    """
    Reinhard tone mapping operator.

    Args:
        hdr_image: Linear HDR image
        key_value: Target average luminance (0.18 is standard)
        white_point: Smallest luminance that will be mapped to white (None = auto)

    Returns:
        Tone-mapped image in [0, 1]
    """

    print("\nApplying Reinhard tone mapping...")

    # Compute luminance
    lum = 0.2126 * hdr_image[:, :, 0] + 0.7152 * \
        hdr_image[:, :, 1] + 0.0722 * hdr_image[:, :, 2]
    lum = np.maximum(lum, 1e-6)

    # Compute log-average luminance
    log_avg_lum = np.exp(np.mean(np.log(lum + 1e-6)))

    # Scale luminance
    scaled_lum = (key_value / log_avg_lum) * lum

    # Apply tone mapping operator
    if white_point is None:
        # Simple global operator
        tone_mapped_lum = scaled_lum / (1.0 + scaled_lum)
    else:
        # With white point
        numerator = scaled_lum * (1.0 + scaled_lum / (white_point ** 2))
        tone_mapped_lum = numerator / (1.0 + scaled_lum)

    # Apply to RGB channels
    scale = tone_mapped_lum / (lum + 1e-6)
    result = hdr_image * scale[:, :, np.newaxis]

    return np.clip(result, 0, 1)


def tone_map_simple_log(hdr_image):
    """Simple logarithmic tone mapping."""

    print("\nApplying logarithmic tone mapping...")

    # Normalize
    hdr_norm = hdr_image / np.percentile(hdr_image, 99)
    hdr_norm = np.clip(hdr_norm, 0, 100)

    # Apply log
    result = np.log1p(hdr_norm * 10) / np.log1p(10 * 100)

    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def hdr_from_jpg_pipeline(data_folder, output_path='results/hdr_from_jpg.jpg'):
    """
    Complete HDR pipeline for JPG images.

    Args:
        data_folder: Folder containing JPG images
        output_path: Where to save final result
    """

    os.makedirs('results', exist_ok=True)

    # Find all JPG images
    jpg_files = sorted([
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ])

    if len(jpg_files) == 0:
        print(f"No JPG files found in {data_folder}")
        return

    print(f"Found {len(jpg_files)} JPG images")
    for f in jpg_files:
        print(f"  - {os.path.basename(f)}")

    # Assume exposure times follow doubling pattern
    # If you know the actual exposure times, replace this
    num_images = len(jpg_files)
    base_time = 1.0
    exposure_times = np.array([base_time / (2 ** i)
                              for i in range(num_images)])

    print(f"\nAssumed exposure times: {exposure_times}")

    # Step 1: Estimate camera response function
    print("\n" + "="*70)
    print("STEP 1: Estimating Camera Response Function")
    print("="*70)
    response_curve = estimate_camera_response_curve(
        jpg_files, exposure_times, num_samples=2000
    )
    plot_response_curve(response_curve)

    # Step 2 & 3: Linearize and merge
    print("\n" + "="*70)
    print("STEP 2-3: Linearizing and Merging Exposures")
    print("="*70)
    hdr_image = merge_exposures_jpg(jpg_files, exposure_times, response_curve)

    # Step 4: Tone mapping
    print("\n" + "="*70)
    print("STEP 4: Tone Mapping")
    print("="*70)

    # Try both tone mapping methods
    ldr_reinhard = tone_map_reinhard(hdr_image, key_value=0.18)
    ldr_log = tone_map_simple_log(hdr_image)

    # Gamma correction for display
    gamma = 0.45
    ldr_reinhard_gamma = np.power(ldr_reinhard, gamma)
    ldr_log_gamma = np.power(ldr_log, gamma)

    # Convert to 8-bit
    result_reinhard = (np.clip(ldr_reinhard_gamma, 0, 1)
                       * 255).astype(np.uint8)
    result_log = (np.clip(ldr_log_gamma, 0, 1) * 255).astype(np.uint8)

    # Save results
    reinhard_path = output_path.replace('.jpg', '_reinhardFull.jpg')
    log_path = output_path.replace('.jpg', '_log.jpg')

    imageio.imwrite(reinhard_path, result_reinhard, quality=95)
    imageio.imwrite(log_path, result_log, quality=95)

    print(f"\nResults saved:")
    print(f"  - Reinhard: {reinhard_path}")
    print(f"  - Log:      {log_path}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Show some original exposures
    for i, idx in enumerate([0, len(jpg_files)//2, -1]):
        img = Image.open(jpg_files[idx])
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original Exposure {idx+1}\n(t={exposure_times[idx]:.4f})',
                             fontsize=10)
        axes[0, i].axis('off')

    # Show HDR results
    axes[1, 0].imshow(result_reinhard)
    axes[1, 0].set_title('HDR - Reinhard Tone Mapping',
                         fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(result_log)
    axes[1, 1].set_title('HDR - Log Tone Mapping',
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # Show middle exposure for comparison
    mid_img = Image.open(jpg_files[len(jpg_files)//2])
    axes[1, 2].imshow(mid_img)
    axes[1, 2].set_title('Reference (Middle Exposure)', fontsize=12)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('results/hdr_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "="*70)
    print("HDR FROM JPG PIPELINE COMPLETE!")
    print("="*70)

    return result_reinhard, result_log


# ============================================================================
# RUN THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    # CHANGE THIS to your actual data folder path
    data_folder = "ex4_additional_exercise_data"

    # Check if folder exists
    if not os.path.exists(data_folder):
        print(f"ERROR: Folder '{data_folder}' not found!")
        print("Please update the 'data_folder' variable to point to your JPG images.")
    else:
        # Run the complete pipeline
        result_reinhard, result_log = hdr_from_jpg_pipeline(
            data_folder,
            output_path='results/hdr_from_jpg_final.jpg'
        )

        print("\n✓ Exercise 9 Complete!")
        print("\nKey Concepts Demonstrated:")
        print("  1. Camera response function estimation from multiple exposures")
        print("  2. Linearization of JPG images (inverting gamma)")
        print("  3. HDR merging with confidence weighting")
        print("  4. Tone mapping (Reinhard & logarithmic)")
        print("\nCheck the 'results/' folder for output images!")
