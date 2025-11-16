import numpy as np
import cv2
import os
from PIL import Image
from PIL.ExifTags import TAGS
import matplotlib.pyplot as plt
from methods import gamma_correction


def get_exposure_time(image_path):
    """Extract exposure time from EXIF data"""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()

        if exif_data is not None:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "ExposureTime":
                    if isinstance(value, tuple):
                        return value[0] / value[1]
                    return float(value)
    except Exception as e:
        print(f"Error reading EXIF from {image_path}: {e}")

    return None


def load_images_from_folder(folder_path):
    """Load all JPEG images from folder"""

    image_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg')):
            image_files.append(os.path.join(folder_path, file))

    image_files.sort()

    print(f"Found {len(image_files)} JPEG images")

    images = []
    exposure_times = []

    for img_path in image_files:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

        exp_time = get_exposure_time(img_path)
        exposure_times.append(exp_time)

        print(f"{os.path.basename(img_path)}: {img.shape}, Exposure: {exp_time}")

    return images, exposure_times, image_files


def estimate_gamma_from_pair(img1, img2, exp_time1, exp_time2, sample_fraction=0.1):
    """Estimate gamma from two images with different exposure times"""

    img1_float = img1.astype(np.float32) / 255.0
    img2_float = img2.astype(np.float32) / 255.0

    # Luminance
    y1 = 0.299 * img1_float[:, :, 0] + 0.587 * \
        img1_float[:, :, 1] + 0.114 * img1_float[:, :, 2]
    y2 = 0.299 * img2_float[:, :, 0] + 0.587 * \
        img2_float[:, :, 1] + 0.114 * img2_float[:, :, 2]

    # Keep middle-range pixels
    mask1 = (y1 > 0.1) & (y1 < 0.9)
    mask2 = (y2 > 0.1) & (y2 < 0.9)
    mask = mask1 & mask2

    # Sample pixels
    valid_indices = np.where(mask)
    num_valid = len(valid_indices[0])
    num_samples = int(num_valid * sample_fraction)

    np.random.seed(42)
    sample_idx = np.random.choice(num_valid, num_samples, replace=False)

    y1_samples = y1[valid_indices[0][sample_idx], valid_indices[1][sample_idx]]
    y2_samples = y2[valid_indices[0][sample_idx], valid_indices[1][sample_idx]]

    # Estimate gamma: γ = log(y1/y2) / log(t1/t2)
    ratio_pixel = y1_samples / (y2_samples + 1e-10)
    ratio_exposure = exp_time1 / exp_time2

    gamma_samples = np.log(ratio_pixel) / np.log(ratio_exposure)
    gamma_samples = gamma_samples[np.isfinite(gamma_samples)]
    gamma = np.median(gamma_samples)

    return gamma


def estimate_gamma_all_pairs(images, exposure_times):
    """
    estimate camera gamma from all adjacent exposures
    """

    print("\nEstimating gamma from image pairs...")

    gammas = []
    for i in range(len(images) - 1):
        gamma = estimate_gamma_from_pair(
            images[i],
            images[i+1],
            exposure_times[i],
            exposure_times[i+1]
        )
        gammas.append(gamma)
        print("pair", i + 1, "gamma", gamma)

    final_gamma = np.median(gammas)

    return final_gamma


def linearize_image(img, gamma):
    """invert gamma to get approximate linear values"""

    inverse_gamma = 1.0 / gamma
    img_float = img.astype(np.float32) / 255.0
    linearized = np.power(img_float, inverse_gamma)

    return linearized


def linearize_all_images(images, gamma):
    """Linearize all images using the estimated gamma"""

    linearized_images = []
    for i, img in enumerate(images):
        lin_img = linearize_image(img, gamma)
        linearized_images.append(lin_img)
        print(f"  Image {i + 1}: linearized")

    return linearized_images


def weight_function(pixel_values):
    """Weight function for HDR merging - favors middle-range values"""

    weights = 1.0 - np.abs(pixel_values - 0.5) * 2.0
    weights = np.maximum(weights, 0.0)
    weights = weights ** 2

    return weights


def merge_hdr(linearized_images, exposure_times):
    """Merge linearized images into HDR using weighted average"""

    print("\nMerging images into HDR...")

    exposure_times = np.array(exposure_times)

    height, width, channels = linearized_images[0].shape
    weighted_sum = np.zeros((height, width, channels), dtype=np.float64)
    weight_sum = np.zeros((height, width, channels), dtype=np.float64)

    for i, (img, exp_time) in enumerate(zip(linearized_images, exposure_times)):
        print(f"  Processing image {i + 1}/{len(linearized_images)}")

        # Approximate original range for weight calculation
        img_original_approx = np.power(img, 0.45)
        weights = weight_function(img_original_approx)

        # Calculate radiance
        radiance = img / (exp_time + 1e-10)

        # Accumulate
        weighted_sum += weights * radiance
        weight_sum += weights

    weight_sum = np.maximum(weight_sum, 1e-10)
    hdr_image = weighted_sum / weight_sum

    print("HDR merge done. Range:", hdr_image.min(), hdr_image.max())

    return hdr_image


def save_result(image, output_path, quality=98):
    """
    Save image to file.
    Handles both [0,1] float and [0,255] uint8 images.
    """

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Convert to uint8
    if image.dtype == np.float32 or image.dtype == np.float64:
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_uint8 = image

    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print(f"Saved to: {output_path}")


def main():
    """Main HDR pipeline"""

    print("=" * 60)
    print("HDR from JPEG Images")
    print("=" * 60)

    # Load images
    print("\n Loading images...")
    folder_path = "ex4_additional_exercise_data"
    images, exposure_times, image_files = load_images_from_folder(folder_path)

    # Estimate gamma
    print("\n Gamma estimation...")
    gamma = estimate_gamma_all_pairs(images, exposure_times)

    # Linearize images
    print("\n Linearizing images...")
    linearized_images = linearize_all_images(images, gamma)

    # Merge into HDR
    print("\n Merging into HDR...")
    hdr_image = merge_hdr(linearized_images, exposure_times)

    # Tone mapping
    print("\n Tone mapping...")
    ldr_result = gamma_correction(
        hdr_image, p_low=0.01, p_high=99.99, gamma=0.35)

    # Save results
    print("\n Saving results...")
    save_result(ldr_result, "results/hdr_final.jpg", quality=98)

    # Visualize
    plt.figure(figsize=(7, 6))
    plt.imshow(ldr_result)
    plt.title('HDR Result (Gamma Tone Mapping)', fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/hdr_result.png', dpi=150, bbox_inches='tight')
    print("Saved visualization: results/hdr_result.png")
    plt.show()

    return hdr_image, ldr_result


# Run the HDR pipeline
hdr_image, ldr_result = main()

print("Output images in results/ ")
