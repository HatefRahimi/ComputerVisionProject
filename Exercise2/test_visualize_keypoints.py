import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_custom_sift_simple(image_path, save_path=None):
    """
    Visualize using YOUR CustomSIFTExtractor (with angle=0 and Hellinger norm)
    """
    from custom_sift_extractor import CustomSIFTExtractor

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use your custom extractor
    extractor = CustomSIFTExtractor()
    kps, desc = extractor.sift.detectAndCompute(gray, None)

    # Apply angle=0
    for kp in kps:
        kp.angle = 0.0
    _, desc = extractor.sift.compute(gray, kps)

    # Apply Hellinger normalization
    desc_hellinger = extractor._hellinger(desc)

    print(f"Image: {image_path}")
    print(f"Keypoints: {len(kps)}")
    print(f"Descriptors shape: {desc_hellinger.shape}")

    # Draw keypoints
    img_with_keypoints = img.copy()
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(img_with_keypoints, (x, y), radius=12,
                   color=(0, 255, 255), thickness=3)
        cv2.circle(img_with_keypoints, (x, y), radius=4,
                   color=(0, 0, 255), thickness=-1)

    # Zoomed region (center 30%)
    h, w = img.shape[:2]
    y1, y2 = int(h * 0.35), int(h * 0.65)
    x1, x2 = int(w * 0.35), int(w * 0.65)
    img_zoom = img_with_keypoints[y1:y2, x1:x2]

    # Convert to RGB
    img_original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_kp_rgb = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
    img_zoom_rgb = cv2.cvtColor(img_zoom, cv2.COLOR_BGR2RGB)

    # Plot with THREE panels
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Original image
    axes[0].imshow(img_original_rgb)
    axes[0].set_title('Original Image', fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # With keypoints
    axes[1].imshow(img_kp_rgb)
    axes[1].set_title(f'Custom SIFT (angle=0, Hellinger) - n={len(kps)}',
                      fontsize=16, fontweight='bold')
    axes[1].axis('off')

    # Zoomed
    axes[2].imshow(img_zoom_rgb)
    axes[2].set_title('Zoomed Region', fontsize=16, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()

    return kps, desc_hellinger


if __name__ == '__main__':
    # Example usage
    image_path = "data/icdar2017-training-color/7-IMG_MAX_10038.jpg"

    print("\n" + "=" * 60)
    print("Custom SIFT")
    print("=" * 60)
    visualize_custom_sift_simple(
        image_path, save_path="custom_sift_keypoints_zoom.png")
