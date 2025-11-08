import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from face_detector import FaceDetector
from face_recognition import FaceRecognizer
from face_recognition import FaceClustering

# 1) point to your folders – pick 2 or 3 distinct people
BASE = os.path.dirname(__file__)  # adjust if needed
SRC = os.path.dirname(BASE)
DATA_ROOT = os.path.dirname(SRC)  # .../data
PEOPLE = ["Alan_Ball", "Manuel_Pellegrini", "Marina_Silva"]
PERSON_DIRS = [os.path.join(DATA_ROOT, "data", "train_data", p) for p in PEOPLE]


# 2) init
detector = FaceDetector(tm_window_size=25, tm_threshold=0.7, aligned_image_size=224)
clustering = FaceClustering(num_clusters=len(PEOPLE), max_iter=50)

# 3) ingest ~10 faces per person
for pid, pdir in enumerate(PERSON_DIRS):
    print(f"\nLoading person #{pid} from {pdir}")
    files = sorted(os.listdir(pdir))[:10]  # take first 10
    for fn in files:
        img = cv2.imread(os.path.join(pdir, fn))
        if img is None: continue
        det = detector.detect_face(img)
        if det:
            clustering.partial_fit(det["aligned"])
        else:
            print("  could not detect in", fn)

print("\nTotal embeddings collected:", clustering.embeddings.shape[0])

# 4) run k-means
obj = clustering.fit()
print("Converged in", len(obj), "iterations. Final objective:", obj[-1])

# 5) print assignments
print("\nCluster assignments:")
idx = 0
for pid, pdir in enumerate(PERSON_DIRS):
    files = sorted(os.listdir(pdir))[:10]
    for fn in files:
        img = cv2.imread(os.path.join(pdir, fn))
        det = detector.detect_face(img)
        if not det:
            continue
        aligned = det["aligned"]
        cluster_id, dists = clustering.predict(aligned)
        print(f"  {fn:15s} → cluster {cluster_id}, distances {dists.round(2)}")
        idx += 1

# 6) plot objective
plt.plot(obj, marker='o')
plt.xlabel("Iteration")
plt.ylabel("k-means objective (sum of squared distances)")
plt.title("Convergence of k-means")
plt.grid(True)
plt.show()
