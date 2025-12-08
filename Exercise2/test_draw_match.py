"""
Visualize SIFT matches between two images using cv2.drawMatches()
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def match_sift_keypoints(image1_path, image2_path, max_matches=50, save_path=None):
    """
    Match SIFT keypoints between two images and visualize with drawMatches

    Parameters:
        image1_path: path to first image
        image2_path: path to second image
        max_matches: maximum number of matches to display
        save_path: optional path to save visualization
    """
    # Read images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None:
        print(f"Error: Could not read {image1_path}")
        return
    if img2 is None:
        print(f"Error: Could not read {image2_path}")
        return

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=1000)

    # Detect keypoints and compute descriptors
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    print(f"Image 1: {len(kp1)} keypoints")
    print(f"Image 2: {len(kp2)} keypoints")

    # Match descriptors using BFMatcher (Brute Force)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"Total matches found: {len(matches)}")
    print(f"Displaying top {min(max_matches, len(matches))} matches")

    # Color-code matches based on distance
    # Green = best, Yellow = medium, Red = worst
    if len(matches) > 0:
        distances = [m.distance for m in matches[:max_matches]]
        min_dist = min(distances)
        max_dist = max(distances)

        # Draw matches with color gradient
        img_matches = img1.copy()
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Create side-by-side image
        img_matches = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        img_matches[:h1, :w1] = img1
        img_matches[:h2, w1:w1+w2] = img2

        # Draw each match with color based on distance
        for match in matches[:max_matches]:
            # Get keypoint coordinates
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            pt2 = tuple(map(int, kp2[match.trainIdx].pt))
            pt2 = (pt2[0] + w1, pt2[1])  # Offset for second image

            # Compute color based on distance (normalize to 0-1)
            if max_dist > min_dist:
                norm_dist = (match.distance - min_dist) / (max_dist - min_dist)
            else:
                norm_dist = 0

            # Color gradient: Green (0) -> Yellow (0.5) -> Red (1)
            if norm_dist < 0.5:
                # Green to Yellow
                r = int(255 * (norm_dist * 2))
                g = 255
                b = 0
            else:
                # Yellow to Red
                r = 255
                g = int(255 * (2 - norm_dist * 2))
                b = 0

            color = (b, g, r)  # OpenCV uses BGR

            # Draw line with varying thickness based on quality
            thickness = 3 if norm_dist < 0.3 else (2 if norm_dist < 0.7 else 1)
            cv2.line(img_matches, pt1, pt2, color, thickness)

            # Draw keypoint circles
            cv2.circle(img_matches, pt1, 5, (255, 0, 0), -1)  # Blue
            cv2.circle(img_matches, pt2, 5, (255, 0, 0), -1)  # Blue
    else:
        # No matches, use default drawMatches
        img_matches = cv2.drawMatches(
            img1, kp1,
            img2, kp2,
            matches[:max_matches],
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    # Convert to RGB for matplotlib
    img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    # Display
    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches_rgb)
    plt.title(f'SIFT Matches (showing {min(max_matches, len(matches))}/{len(matches)} matches)\n' +
              'Green=Best matches, Yellow=Medium, Red=Worst',
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()

    # Print statistics about match quality
    if len(matches) > 0:
        distances = [m.distance for m in matches[:max_matches]]
        print(f"\nMatch quality statistics (top {max_matches}):")
        print(f"  Best match distance: {min(distances):.2f}")
        print(f"  Worst match distance: {max(distances):.2f}")
        print(f"  Average distance: {np.mean(distances):.2f}")

    return matches, kp1, kp2


def match_custom_sift(image1_path, image2_path, max_matches=50, save_path=None):
    """
    Match using YOUR CustomSIFTExtractor (angle=0, Hellinger normalization)
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

    print(f"Custom SIFT - Image 1: {len(kp1)} keypoints")
    print(f"Custom SIFT - Image 2: {len(kp2)} keypoints")

    # Match with L2 distance
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"Total matches: {len(matches)}")
    print(f"Displaying top {min(max_matches, len(matches))} matches")

    # Color-code matches based on distance
    if len(matches) > 0:
        distances = [m.distance for m in matches[:max_matches]]
        min_dist = min(distances)
        max_dist = max(distances)

        # Create side-by-side image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        img_matches = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        img_matches[:h1, :w1] = img1
        img_matches[:h2, w1:w1+w2] = img2

        # Draw each match with color gradient
        for match in matches[:max_matches]:
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            pt2 = tuple(map(int, kp2[match.trainIdx].pt))
            pt2 = (pt2[0] + w1, pt2[1])

            # Normalize distance to 0-1
            if max_dist > min_dist:
                norm_dist = (match.distance - min_dist) / (max_dist - min_dist)
            else:
                norm_dist = 0

            # Green -> Yellow -> Red gradient
            if norm_dist < 0.5:
                r = int(255 * (norm_dist * 2))
                g = 255
                b = 0
            else:
                r = 255
                g = int(255 * (2 - norm_dist * 2))
                b = 0

            color = (b, g, r)
            thickness = 3 if norm_dist < 0.3 else (2 if norm_dist < 0.7 else 1)

            cv2.line(img_matches, pt1, pt2, color, thickness)
            cv2.circle(img_matches, pt1, 5, (255, 0, 0), -1)
            cv2.circle(img_matches, pt2, 5, (255, 0, 0), -1)
    else:
        img_matches = cv2.drawMatches(
            img1, kp1,
            img2, kp2,
            matches[:max_matches],
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches_rgb)
    plt.title(f'Custom SIFT Matches (angle=0, Hellinger) - {min(max_matches, len(matches))}/{len(matches)} matches\n' +
              'Green=Best matches, Yellow=Medium, Red=Worst',
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()

    if len(matches) > 0:
        distances = [m.distance for m in matches[:max_matches]]
        print(f"\nMatch quality:")
        print(f"  Best: {min(distances):.2f}")
        print(f"  Worst: {max(distances):.2f}")
        print(f"  Average: {np.mean(distances):.2f}")

    return matches, kp1, kp2


def compare_same_vs_different_writer():
    """
    Compare matches between:
    1. Same writer, different documents → many matches expected
    2. Different writers → few matches expected
    """
    # You'll need to adjust these paths to your actual data
    base_path = "data/icdar2017-training-color"

    # Example: two documents from writer 1
    # Replace with actual filenames from your dataset
    writer1_doc1 = os.path.join(base_path, "1000-IMG_MAX_116390.png")
    # Replace with actual file
    writer1_doc2 = os.path.join(base_path, "1000-IMG_MAX_116391.png")

    # Example: document from writer 2
    # Replace with actual file
    writer2_doc1 = os.path.join(base_path, "1001-IMG_MAX_123456.png")

    print("=" * 70)
    print("Scenario 1: SAME WRITER, Different Documents")
    print("=" * 70)
    print("Expected: MANY matches (similar writing style)\n")

    if os.path.exists(writer1_doc1) and os.path.exists(writer1_doc2):
        match_sift_keypoints(
            writer1_doc1,
            writer1_doc2,
            max_matches=100,
            save_path="matches_same_writer.png"
        )
    else:
        print(f"Files not found. Please update paths.")

    print("\n" + "=" * 70)
    print("Scenario 2: DIFFERENT WRITERS")
    print("=" * 70)
    print("Expected: FEW matches (different writing styles)\n")

    if os.path.exists(writer1_doc1) and os.path.exists(writer2_doc1):
        match_sift_keypoints(
            writer1_doc1,
            writer2_doc1,
            max_matches=100,
            save_path="matches_different_writers.png"
        )
    else:
        print(f"Files not found. Please update paths.")


if __name__ == '__main__':
    # Example 1: Compare image to itself (sanity check)
    print("=" * 70)
    print("Example 1: SANITY CHECK - Compare Image to Itself")
    print("=" * 70)
    print("Expected: ALL keypoints match with distance ≈ 0\n")

    image1 = "data/icdar2017-training-color/7-IMG_MAX_10038.jpg"

    if os.path.exists(image1):
        print("Standard SIFT (self-comparison):")
        match_sift_keypoints(image1, image1, max_matches=100,
                             save_path="sift_matches_self.png")

        print("\nCustom SIFT (self-comparison):")
        match_custom_sift(image1, image1, max_matches=100,
                          save_path="custom_sift_matches_self.png")
    else:
        print(f"Please update image path: {image1}")

    # Example 2: Compare two different images from same writer
    print("\n" + "=" * 70)
    print("Example 2: Two Different Images (update paths as needed)")
    print("=" * 70)

    # Replace these with your actual image paths
    image2 = "data/icdar2017-training-color/1000-IMG_MAX_116391.png"  # Same writer

    print("\nStandard SIFT Matching:")
    print("-" * 70)
    if os.path.exists(image1) and os.path.exists(image2):
        match_sift_keypoints(image1, image2, max_matches=50,
                             save_path="sift_matches.png")
    else:
        print(f"Please update image paths in the script!")
        print(f"Looking for:")
        print(f"  - {image1}")
        print(f"  - {image2}")

    print("\nCustom SIFT Matching (angle=0, Hellinger):")
    print("-" * 70)
    if os.path.exists(image1) and os.path.exists(image2):
        match_custom_sift(image1, image2, max_matches=50,
                          save_path="custom_sift_matches.png")
    else:
        print("Please update image paths!")

    # Uncomment to compare same vs different writers:
    # print("\n3. Compare Same vs Different Writers:")
    # print("-" * 70)
    # compare_same_vs_different_writer()
