"""
Visualize Custom SIFT matches with each line a different color
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def match_custom_sift_colorful(image1_path, image2_path, max_matches=50, save_path=None):
    """
    Match using YOUR CustomSIFTExtractor with EACH LINE a different random color
    """
    from custom_sift_extractor import CustomSIFTExtractor

    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print("Error reading images")
        return

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Use your custom extractor
    extractor = CustomSIFTExtractor()

    # Extract from both images
    kp1, _ = extractor.sift.detectAndCompute(gray1, None)
    for kp in kp1:
        kp.angle = 0.0
    _, desc1 = extractor.sift.compute(gray1, kp1)
    desc1 = extractor._hellinger(desc1)

    kp2, _ = extractor.sift.detectAndCompute(gray2, None)
    for kp in kp2:
        kp.angle = 0.0
    _, desc2 = extractor.sift.compute(gray2, kp2)
    desc2 = extractor._hellinger(desc2)

    print(f"Image 1: {len(kp1)} keypoints")
    print(f"Image 2: {len(kp2)} keypoints")

    # Match with L2 distance
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"Total matches: {len(matches)}")
    print(
        f"Displaying {min(max_matches, len(matches))} matches with random colors")

    # Create side-by-side image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img_matches = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    img_matches[:h1, :w1] = img1
    img_matches[:h2, w1:w1+w2] = img2

    # Draw each match with a DIFFERENT random color
    np.random.seed(42)  # For reproducible colors
    for match in matches[:max_matches]:
        pt1 = tuple(map(int, kp1[match.queryIdx].pt))
        pt2 = tuple(map(int, kp2[match.trainIdx].pt))
        pt2 = (pt2[0] + w1, pt2[1])  # Offset for second image

        color = (
            int(np.random.randint(0, 255)),  # B
            int(np.random.randint(0, 255)),  # G
            int(np.random.randint(0, 255))   # R
        )

        # Draw line
        cv2.line(img_matches, pt1, pt2, color, thickness=2)

        # Draw keypoint circles
        cv2.circle(img_matches, pt1, 5, color, -1)
        cv2.circle(img_matches, pt2, 5, color, -1)

    # Convert to RGB for matplotlib
    img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    # Display
    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches_rgb)
    plt.title(f'Custom SIFT Matches - {min(max_matches, len(matches))}/{len(matches)} matches ',
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()

    # Print statistics
    if len(matches) > 0:
        distances = [m.distance for m in matches[:max_matches]]
        print(f"\nMatch quality:")
        print(f"  Best distance: {min(distances):.2f}")
        print(f"  Worst distance: {max(distances):.2f}")
        print(f"  Average distance: {np.mean(distances):.2f}")

    return matches, kp1, kp2


if __name__ == '__main__':
    print("=" * 70)
    print("Custom SIFT - Compare Image to Itself")
    print("=" * 70)
    print("Expected: ALL keypoints match with distance ≈ 0")
    print("Each line will be a different random color\n")

    # Update this path to your image
    image_path = "data/icdar2017-training-color/7-IMG_MAX_10038.jpg"

    if os.path.exists(image_path):
        match_custom_sift_colorful(
            image_path,
            image_path,  # Same image (self-comparison)
            max_matches=100,
            save_path="custom_sift_colorful_matches.png"
        )
    else:
        print(f"Error: Image not found at {image_path}")
        print("Please update the image_path in the script!")
