# import os, cv2, numpy as np
#
# from face_detector import FaceDetector
# from face_recognition import FaceRecognizer
#
# # 1) Calculate the real data folder
# # HERE = os.path.dirname(__file__)           # .../data/src/cvproj_exc
# # SRC  = os.path.dirname(HERE)               # .../data/src
# # DATA = os.path.dirname(SRC)                # .../data
# #
# # print("Loading images from:", DATA)
# # print("Contents:", os.listdir(DATA))
#
# # 2) Instantiate recognizer
# detector = FaceDetector(tm_window_size=25, tm_threshold=0.7, aligned_image_size=224)
# rec = FaceRecognizer(num_neighbours=1, max_distance=1.3, min_prob=0.3)
#
# DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#
# # TRAINING
# for fname, label in [
#     ("charlize.jpg",     "charlize"),
#     ("Charlize (2).jpg", "charlize"),
#     ("kristian.jpg",     "kristian"),
# ]:
#     path = os.path.join(DATA_DIR, fname)
#     img  = cv2.imread(path)
#     if img is None:
#         print("Missing:", path); continue
#
#     # 1) Detect & align the face in this single image
#     det = detector.detect_face(img)
#     if not det:
#         print("Couldn’t detect face in:", path); continue
#     aligned = det["aligned"]    # guaranteed to be 224×224
#
#     # 2) Now partial_fit on that aligned patch
#     rec.partial_fit(aligned, label)
#
# print("Gallery labels:", rec.labels)
# print("Gallery embeddings shape:", rec.embeddings.shape)
#
#
# # TESTING
# for fname in ["charlize.jpg","kris.jpg","angelina.jpg"]:
#     path = os.path.join(DATA_DIR, fname)
#     img  = cv2.imread(path)
#     det = detector.detect_face(img)
#     aligned = det["aligned"]
#
#     # Get the probe embedding
#     probe_emb = rec.facenet.predict(aligned).flatten()
#
#     # Compute all distances to your gallery
#     dists = np.linalg.norm(rec.embeddings - probe_emb, axis=1)
#     for lbl, dist in zip(rec.labels, dists):
#         print(f"  Comparing to {lbl:10s} → dist = {dist:.3f}")
#     print("\n")
#     print(f"  → min distance = {dists.min():.3f}\n")
#     if img is None:
#         print("Missing:", path); continue
#
#     det = detector.detect_face(img)
#     if not det:
#         print("No face in:", path); continue
#     aligned = det["aligned"]
#
#     label, conf = rec.predict(aligned)
#     print(f"{fname:>12s} → {label!r}, conf={conf:.2f}")


import os
import cv2
import numpy as np
from collections import defaultdict

from face_detector import FaceDetector
from face_recognition import FaceRecognizer

# ------------------------------------------------------------------
# 1) Locate your train/test folders (adjust to your layout)
# ------------------------------------------------------------------
# file is .../data/src/cvproj_exc/train_and_eval.py
HERE = os.path.dirname(__file__)
SRC = os.path.dirname(HERE)  # .../data/src
DATA_ROOT = os.path.dirname(SRC)  # .../data

# TRAIN_DIR = os.path.join(DATA_ROOT, "data", "train_data")
# TEST_DIR = os.path.join(DATA_ROOT, "data", "test_data")
#
# print("Train dir:", TRAIN_DIR)
# print("Test  dir:", TEST_DIR)
#
# # ------------------------------------------------------------------
# # 2) Initialize detector & recognizer
# # ------------------------------------------------------------------
detector = FaceDetector(tm_window_size=25, tm_threshold=0.7, aligned_image_size=224)
recognizer = FaceRecognizer(num_neighbours=5)

# # ------------------------------------------------------------------
# # 3) Build the gallery from train_data
# # ------------------------------------------------------------------
# print("\n=== TRAINING ===")
# for person in os.listdir(TRAIN_DIR):
#     person_dir = os.path.join(TRAIN_DIR, person)
#     if not os.path.isdir(person_dir):
#         continue
#
#     print(f"--> {person}")
#     for img_name in os.listdir(person_dir):
#         img_path = os.path.join(person_dir, img_name)
#         img = cv2.imread(img_path)
#         if img is None:
#             continue
#
#         # detect + align
#         det = detector.detect_face(img)
#         if det is None:
#             print("   [no face] ", img_name)
#             continue
#         aligned = det["aligned"]
#
#         # partial_fit with color+gray embeddings
#         recognizer.partial_fit(aligned, person)
#
# print("Gallery size:", len(recognizer.labels),
#       "embeddings:", recognizer.embeddings.shape)
#
# # ------------------------------------------------------------------
# # 4) Evaluate on test_data
# # ------------------------------------------------------------------
# print("\n=== TESTING ===")
# results = defaultdict(lambda: {"correct": 0, "total": 0})
#
# for person in os.listdir(TEST_DIR):
#     person_dir = os.path.join(TEST_DIR, person)
#     if not os.path.isdir(person_dir):
#         continue
#
#     print(f"** {person} **")
#     for img_name in os.listdir(person_dir):
#         img_path = os.path.join(person_dir, img_name)
#         img = cv2.imread(img_path)
#         if img is None:
#             continue
#
#         det = detector.detect_face(img)
#         if det is None:
#             print("   [no face] ", img_name)
#             continue
#         aligned = det["aligned"]
#
#         # predict
#         pred_label, p, d = recognizer.predict(aligned)
#         correct = (pred_label == person)
#         results[person]["total"] += 1
#         results[person]["correct"] += int(correct)
#
#         status = "OK" if correct else "ERR"
#         print(f"   {img_name:15s} → pred={pred_label!r:<12s}"
#               f" p={p:.2f} d={d:.3f} [{status}]")

# ------------------------------------------------------------------
# 5) Print per‐class accuracy
# ------------------------------------------------------------------
# print("\n=== ACCURACY ===")
# for person, stats in results.items():
#     total = stats["total"]
#     correct = stats["correct"]
#     acc = correct / total if total > 0 else 0.0
#     print(f"{person:15s}: {correct:3d}/{total:3d}  acc={acc:.2%}")

UNKNOWN_DIR = os.path.join(DATA_ROOT, "data", "unknown_data")
KNOWN_DIR = os.path.join(DATA_ROOT, "data", "train_data")
KNOWN_DIR2 = os.path.join(DATA_ROOT, "data", "test_data")
print(f"\n=== OPEN-SET EVAL ===")
print("Unknown dir:", UNKNOWN_DIR)


# 1) TRAIN the gallery
for person in os.listdir(KNOWN_DIR2):
    pdir = os.path.join(KNOWN_DIR2, person)
    for fn in os.listdir(pdir):
        img = cv2.imread(os.path.join(pdir, fn))
        det = detector.detect_face(img)
        if det:
            recognizer.partial_fit(det["aligned"], person)

print("Gallery size after training:", recognizer.embeddings.shape[0])

# 2) COLLECT known_faces (aligned)
known_faces = []
for person in os.listdir(KNOWN_DIR):
    pdir = os.path.join(KNOWN_DIR, person)
    for fn in os.listdir(pdir):
        img = cv2.imread(os.path.join(pdir, fn))
        det = detector.detect_face(img)
        if det:
            known_faces.append((person, det["aligned"]))

# 3) COLLECT unknown_faces (aligned)
unknown_faces = []
for person in os.listdir(UNKNOWN_DIR):
    pdir = os.path.join(UNKNOWN_DIR, person)
    for fn in os.listdir(pdir):
        img = cv2.imread(os.path.join(pdir, fn))
        det = detector.detect_face(img)
        if det:
            unknown_faces.append(det["aligned"])

# 4) NOW you can call predict
known_ds = [recognizer.predict(f)[2] for (_, f) in known_faces]
unknown_ds = [recognizer.predict(f)[2] for f in unknown_faces]
print("Known dists  :", sorted(known_ds)[:3], "...", sorted(known_ds)[-3:])
print("Unknown dists:", sorted(unknown_ds)[:3], "...", sorted(unknown_ds)[-3:])
