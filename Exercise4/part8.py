import os
import numpy as np
import rawpy
import imageio
from scipy.ndimage import gaussian_filter
import time

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)


def process_raw(raw_path, output_filename="output_custom.jpg", quality=99):
    start_time = time.time()
    print(f"Processing {os.path.basename(raw_path)}...")

    # Step 1: Read and demosaic using rawpy
    with rawpy.imread(raw_path) as raw:
        # Get a basic RGB image using rawpy's demosaicing
        # This avoids Bayer pattern issues
        rgb = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  # High quality
            use_camera_wb=False,  # white balance
            use_auto_wb=False,
            no_auto_bright=True,
            output_bps=16,  # 16-bit for better processing
            gamma=(1, 1),  # Linear output (no gamma)
            output_color=rawpy.ColorSpace.raw  # Stay in camera space
        )

    # Convert to float32 for processing
    rgb = rgb.astype(np.float32) / 65535.0

    print("Applying custom processing pipeline...")

    # Step 2: Noise reduction (optional, mild)
    print("Denoising...")
    for c in range(3):
        rgb[:, :, c] = gaussian_filter(rgb[:, :, c], sigma=0.5)

    # Step 3: White balance (gray world)
    print("White balancing...")
    for c in range(3):
        mean_val = np.mean(rgb[:, :, c])
        if mean_val > 0:
            rgb[:, :, c] = rgb[:, :, c] / mean_val * 0.4  # Target gray

    # Step 4: Tone mapping (adaptive logarithmic)
    print("Tone mapping...")
    # Calculate luminance
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    lum = np.maximum(lum, 1e-6)

    # Adaptive tone mapping
    lum_avg = np.exp(np.mean(np.log(lum)))
    key_value = 0.18
    lum_scaled = (key_value / lum_avg) * lum

    # Compress dynamic range
    lum_compressed = lum_scaled / (1 + lum_scaled)

    # Apply to RGB channels
    scale = lum_compressed / lum
    rgb = rgb * scale[:, :, np.newaxis]

    # Step 5: Local contrast enhancement
    print("Enhancing local contrast...")
    # Simple unsharp masking
    for c in range(3):
        blurred = gaussian_filter(rgb[:, :, c], sigma=10)
        detail = rgb[:, :, c] - blurred
        rgb[:, :, c] = rgb[:, :, c] + 0.3 * detail

    # Step 6: Color enhancement
    print("Color grading...")
    # Increase saturation slightly
    gray = np.mean(rgb, axis=2, keepdims=True)
    rgb = gray + 1.2 * (rgb - gray)

    # Step 7: Final adjustments
    # Ensure proper range
    rgb = np.clip(rgb, 0, 1)

    # Apply S-curve for better contrast
    rgb = rgb * rgb * (3 - 2 * rgb)

    # Apply gamma for display
    gamma = 1.0 / 2.2
    rgb = np.power(rgb, gamma)

    # Step 8: Convert to 8-bit and save
    rgb_8bit = (rgb * 255).astype(np.uint8)

    # --- Save into results/ folder ---
    output_path = os.path.join(results_dir, output_filename)
    imageio.imwrite(output_path, rgb_8bit, quality=quality)

    elapsed_time = time.time() - start_time
    print(f"Processing complete in {elapsed_time:.2f} seconds")
    print(f"Image saved to: {output_path}")
    print(f"Image shape: {rgb_8bit.shape}")

    return rgb_8bit


raw_file = "exercise_4_data/06/03.CR3"

# Try the custom processing
process_raw(raw_file, "output_custom.jpg", quality=99)
