import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_jpg_stack(folder):

    jpg_files = sorted(
        fname for fname in os.listdir(folder)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if len(jpg_files) == 0:
        raise FileNotFoundError(
            f"No JPG/PNG images found in folder: {folder}"
        )

    for fname in jpg_files:
        print("  ", fname)

    # 2) Hard-code exposure times (brightest to darkest)
    exposure_times = np.array([
        13.0,      # Longest exposure (brightest)
        6.0,
        3.0,
        2.0,
        1.0,
        0.5,
        0.2,
        0.1,
        0.05,
        0.025,
        0.0125,
        0.00625
    ], dtype=np.float32)

    if len(jpg_files) != len(exposure_times):
        raise ValueError(
            f"Number of images ({len(jpg_files)}) does not match "
            f"number of exposure times ({len(exposure_times)})."
        )

    images = []
    for fname in jpg_files:
        path = os.path.join(folder, fname)

        img = Image.open(path).convert("RGB")
        img_np = np.asarray(img, dtype=np.float32) / 255.0

        images.append(img_np)

    # Sanity-check shapes
    heights = [im.shape[0] for im in images]
    widths = [im.shape[1] for im in images]
    if not (len(set(heights)) == 1 and len(set(widths)) == 1):
        print("Warning: images have different resolutions!")
        print("Heights:", heights)
        print("Widths:", widths)

    return images, exposure_times, jpg_files


if __name__ == "__main__":
    data_folder = "ex4_additional_exercise_data"

    images, exposure_times, filenames = load_jpg_stack(data_folder)

    print("Number of images:", len(images))
    print("Image shape:", images[0].shape)
    print("Exposure times:", exposure_times)

    # Quick visualization of first 3 exposures
    num_show = min(3, len(images))
    plt.figure(figsize=(4 * num_show, 4))
    for i in range(num_show):
        plt.subplot(1, num_show, i + 1)
        plt.imshow(images[i])
        plt.title(f"{filenames[i]}\n t = {exposure_times[i]} s")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
