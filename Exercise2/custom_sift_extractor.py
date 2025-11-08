# custom_sift_extractor.py
import os
import shlex
import gzip
import cv2
import numpy as np
import _pickle as cPickle
from typing import List, Tuple
from tqdm import tqdm


class CustomSIFTExtractor:
    """
    Part (e): Custom SIFT extractor for binary images.
    """

    KNOWN_EXTS = [".png", ".PNG", ".jpg", ".JPG", ".jpeg",
                  ".JPEG", ".tif", ".TIF", ".tiff", ".TIFF"]

    def __init__(self,
                 use_clahe: bool = True,
                 dense_step: int = 8,
                 dense_size: int = 8,
                 sift_nfeatures: int = 4000,
                 sift_contrast: float = 0.005,
                 sift_edge: int = 10,
                 sift_sigma: float = 1.6):
        self.use_clahe = use_clahe
        self.dense_step = dense_step
        self.dense_size = dense_size
        self.sift = cv2.SIFT_create(
            nfeatures=sift_nfeatures,
            contrastThreshold=sift_contrast,
            edgeThreshold=sift_edge,
            sigma=sift_sigma
        )
        self._clahe = None
        if self.use_clahe:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def compute(self, img_path: str) -> np.ndarray:
        """
        Return Hellinger-normalized SIFT descriptors (N,128) float32.
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((0, 128), np.float32)

        if self._clahe is not None:
            img = self._clahe.apply(img)

        kps, desc = self.sift.detectAndCompute(img, None)
        if desc is None or len(desc) == 0:
            kps = self._dense_keypoints(
                img, step=self.dense_step, size=self.dense_size)
            _, desc = self.sift.compute(img, kps)
            if desc is None or len(desc) == 0:
                return np.zeros((0, 128), np.float32)

        # force orientation = 0 and recompute
        for kp in kps:
            kp.angle = 0.0
        _, desc = self.sift.compute(img, kps)
        if desc is None or len(desc) == 0:
            return np.zeros((0, 128), np.float32)

        return self._hellinger(desc)

    def build_for_split(self, labels_file: str, out_folder: str, search_dirs: List[str]):
        """
        Build & save descriptors for all basenames in labels_file.
        Returns: (produced_files, nonempty_files, total_rows)
        """
        os.makedirs(out_folder, exist_ok=True)
        pairs = self._read_label_basenames(labels_file)

        produced = 0
        nonempty = 0
        total_rows = 0
        missing = 0

        for base, _ in tqdm(pairs, desc=f"Part (e): building descriptors → {out_folder}"):
            out_path = os.path.join(out_folder, base + "_SIFT_patch_pr.pkl.gz")
            if os.path.exists(out_path):
                # count stats and skip re-extraction
                try:
                    with gzip.open(out_path, 'rb') as f:
                        arr = cPickle.load(f, encoding='latin1')
                    produced += 1
                    if arr is not None and getattr(arr, "shape", (0, 0))[0] > 0:
                        nonempty += 1
                        total_rows += arr.shape[0]
                except Exception:
                    pass
                continue

            img_path = self._resolve_image_path(base, search_dirs)
            if not img_path:
                missing += 1
                continue

            desc = self.compute(img_path)
            with gzip.open(out_path, 'wb') as f:
                cPickle.dump(desc.astype(np.float32), f, -1)

            produced += 1
            if desc is not None and len(desc) > 0:
                nonempty += 1
                total_rows += desc.shape[0]

        if missing > 0:
            print(
                f"[Part (e)] Warning: {missing} images not found in {search_dirs}.")
        print(f"[Part (e)] Wrote/kept {produced} descriptor files "
              f"({nonempty} non-empty, {total_rows} total SIFT vectors).")
        return produced, nonempty, total_rows

    # Helpers

    def _dense_keypoints(self, img, step=8, size=8):
        h, w = img.shape[:2]
        return [cv2.KeyPoint(float(x), float(y), _size=float(size))
                for y in range(0, h, step) for x in range(0, w, step)]

    def _hellinger(self, desc: np.ndarray) -> np.ndarray:
        if desc is None or len(desc) == 0:
            return np.zeros((0, 128), np.float32)
        desc = desc.astype(np.float32)
        l1 = np.sum(np.abs(desc), axis=1, keepdims=True)
        l1[l1 == 0] = 1.0
        desc = desc / l1
        return np.sign(desc) * np.sqrt(np.abs(desc) + 1e-12)

    def _resolve_image_path(self, basename: str, search_dirs: List[str]) -> str:
        for d in search_dirs:
            for ext in self.KNOWN_EXTS:
                p = os.path.join(d, basename + ext)
                if os.path.exists(p):
                    return p
        return ""

    def _read_label_basenames(self, labels_file: str) -> List[Tuple[str, str]]:
        pairs = []
        with open(labels_file, "r") as f:
            for line in f:
                s = shlex.split(line.strip())
                if len(s) < 2:
                    continue
                base, lab = s[0], s[1]
                for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ocvmb', '.csv']:
                    if base.endswith(p):
                        base = base[:-len(p)]
                pairs.append((base, lab))
        return pairs
