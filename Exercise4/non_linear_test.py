import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS

# ---------- Helper: read exposure time from EXIF ----------


def get_exposure_time(image_path):
    """
    Extract ExposureTime from EXIF.
    Returns a float in seconds, or None if not found.
    """
    img = Image.open(image_path)
    exif_data = img._getexif()
    if exif_data is None:
        return None

    exposure_tag = None
    for tag_id, tag_name in TAGS.items():
        if tag_name == "ExposureTime":
            exposure_tag = tag_id
            break

    if exposure_tag is None or exposure_tag not in exif_data:
        return None

    value = exif_data[exposure_tag]
    # usually stored as (num, den)
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        return float(num) / float(den)
    else:
        return float(value)


# ---------- Folder with JPEGs ----------
folder = "ex4_additional_exercise_data"

jpeg_files = [
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg"))
]

# Read exposure times
exposure_times = []
for f in jpeg_files:
    t = get_exposure_time(f)
    if t is None:
        raise RuntimeError(f"No ExposureTime in EXIF for {f}")
    exposure_times.append(t)

exposure_times = np.array(exposure_times, dtype=float)

# Sort by exposure time
order = np.argsort(exposure_times)
exposure_times = exposure_times[order]
jpeg_files = [jpeg_files[i] for i in order]

# ---------- Compute mean luminance (single value per image) ----------
mean_luminance = []

for path in jpeg_files:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0  # [0,1]

    R = img_rgb[..., 0]
    G = img_rgb[..., 1]
    B = img_rgb[..., 2]

    # luminance (you could also just use img_rgb.mean())
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    mean_luminance.append(Y.mean())

mean_luminance = np.array(mean_luminance)

# ---------- Plot: one curve ----------
plt.figure(figsize=(8, 6))
plt.plot(exposure_times, mean_luminance, "-o")

plt.xlabel("Exposure time (s)")
plt.ylabel("Mean JPEG luminance (0–1)")
plt.title("JPEG Non-Linearity: Mean Luminance vs. Exposure Time")
plt.grid(True, ls="--", lw=0.5)
plt.tight_layout()
plt.show()
