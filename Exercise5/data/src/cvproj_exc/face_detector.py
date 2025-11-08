import cv2
import numpy as np
from mtcnn import MTCNN


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=25, tm_threshold=0.2, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

        # TODO: Specify all parameters for template matching.
        self.tm_window = tm_window_size
        self.tm_threshold = tm_threshold

    # TODO: Track a face in a new image using template matching.
    def track_face(self, image):
        # 1) First frame
        if self.reference is None:
            det = self.detect_face(image)
            if det is None:
                return None
            self.reference = det
            return det

        # 2) Try fast template match
        x, y, w, h = self.reference["rect"]
        H, W = image.shape[:2]
        m = self.tm_window
        x0, y0 = max(0, x - m), max(0, y - m)
        x1, y1 = min(W, x + w + m), min(H, y + h + m)
        search = image[y0:y1, x0:x1]

        th, tw = self.reference["aligned"].shape[:2]
        hs, ws = search.shape[:2]
        # if search is still too small, skip to fallback
        if hs < th or ws < tw:
            det = self.detect_face(image)
            if det is None:
                return None
            self.reference = det
            return det

        res = cv2.matchTemplate(search, self.reference["aligned"], cv2.TM_CCOEFF_NORMED)
        _, max_val, _, (dx, dy) = cv2.minMaxLoc(res)

        if max_val < self.tm_threshold:
            # 3) Fallback
            det = self.detect_face(image)
            if det is None:
                return None
            self.reference = det
            return det

        # 4) Update on good match
        new_x, new_y = x0 + dx, y0 + dy
        kps = self.reference["keypoints"]

        # 5) Pose‐normalize by recomputing similarity warp
        aligned = self._pose_normalize_and_crop(image, (new_x, new_y, w, h),  self.reference["keypoints"])

        # 6) Store and return
        self.reference = {
            "rect": (new_x, new_y, w, h),
            "keypoints": kps,
            "aligned": aligned,
            "response": max_val
        }
        return self.reference

    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        if not (
            detections := self.detector.detect_faces(image, threshold_pnet=0.85, threshold_rnet=0.9)
        ):
            self.reference = None
            return None

        # Select face with the largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]
        keypoints = detections[largest_detection]["keypoints"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0, "keypoints": keypoints}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(
            self.crop_face(image, face_rect),
            dsize=(self.aligned_image_size, self.aligned_image_size),
        )

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]

    def _pose_normalize_and_crop(self, image, face_rect, keypoints):
        # 1) Source eye positions
        left_eye = np.array(keypoints["left_eye"], dtype=np.float32)
        right_eye = np.array(keypoints["right_eye"], dtype=np.float32)

        # 2) Target eye positions in the output
        s = self.aligned_image_size
        dst_left = np.array([0.3 * s, 0.4 * s], dtype=np.float32)
        dst_right = np.array([0.7 * s, 0.4 * s], dtype=np.float32)

        # 3) Estimate 2×3 similarity transform
        src = np.stack([left_eye, right_eye])
        dst = np.stack([dst_left, dst_right])
        # OpenCV wants shape (N,1,2)
        src = src.reshape(-1, 1, 2)
        dst = dst.reshape(-1, 1, 2)
        M, _ = cv2.estimateAffinePartial2D(src, dst)

        # 4) Warp the full image to the canonical frame
        warped = cv2.warpAffine(
            image, M, (s, s),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        # 5) Return the aligned crop (already s×s)
        return warped
