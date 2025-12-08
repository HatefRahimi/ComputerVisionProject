"""
Test SIFT rotation invariance - compare original vs rotated image
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def rotate_image(img, angle):
    """Rotate image by angle degrees"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    return rotated


def compare_custom_sift_rotation(image_path, rotation_angle=15):
    """
    Compare YOUR custom SIFT (angle=0) on original vs rotated
    """
    from custom_sift_extractor import CustomSIFTExtractor

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Rotate
    img_rotated = rotate_image(img, rotation_angle)
    gray_rotated = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)

    # Extract with custom extractor
    extractor = CustomSIFTExtractor()

    # Original
    kps_orig, _ = extractor.sift.detectAndCompute(gray, None)
    for kp in kps_orig:
        kp.angle = 0.0
    _, desc_orig = extractor.sift.compute(gray, kps_orig)
    desc_orig = extractor._hellinger(desc_orig)

    # Rotated
    kps_rot, _ = extractor.sift.detectAndCompute(gray_rotated, None)
    for kp in kps_rot:
        kp.angle = 0.0
    _, desc_rot = extractor.sift.compute(gray_rotated, kps_rot)
    desc_rot = extractor._hellinger(desc_rot)

    print(f"\nCustom SIFT (angle=0):")
    print(f"Original: {len(kps_orig)} keypoints")
    print(f"Rotated {rotation_angle}°: {len(kps_rot)} keypoints")

    # Draw - SAME STYLE as visualize_sift_simple
    img_kp_orig = img.copy()
    img_kp_rot = img_rotated.copy()

    for kp in kps_orig:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        # Yellow circle
        cv2.circle(img_kp_orig, (x, y), radius=12,
                   color=(0, 255, 255), thickness=3)
        # Red center dot
        cv2.circle(img_kp_orig, (x, y), radius=4,
                   color=(0, 0, 255), thickness=-1)

    for kp in kps_rot:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        # Yellow circle
        cv2.circle(img_kp_rot, (x, y), radius=12,
                   color=(0, 255, 255), thickness=3)
        # Red center dot
        cv2.circle(img_kp_rot, (x, y), radius=4,
                   color=(0, 0, 255), thickness=-1)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(img_kp_orig, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Custom SIFT angle=0 (n={len(kps_orig)})',
                         fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(
        f'Rotated {rotation_angle}°', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(img_kp_rot, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Custom SIFT angle=0 (n={len(kps_rot)})',
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('custom_sift_rotation_test.png', dpi=150, bbox_inches='tight')
    print("Saved to: custom_sift_rotation_test.png")
    plt.show()


if __name__ == '__main__':
    image_path = "data/icdar2017-training-color/7-IMG_MAX_10038.jpg"

    print("\n" + "=" * 60)
    print("Test 2: Custom SIFT with angle=0 (NOT rotation-invariant)")
    print("=" * 60)
    compare_custom_sift_rotation(image_path, rotation_angle=15)
