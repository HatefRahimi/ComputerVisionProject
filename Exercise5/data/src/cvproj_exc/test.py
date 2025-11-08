import argparse

import cv2
import numpy as np

from config import Config, ReIdMode, enum_choices
from face_recognition import FaceRecognizer
from face_detector import FaceDetector
from face_recognition import FaceClustering


# from face_recognition import FaceClustering, FaceRecognizer


# The test module of the face recognition system. This comprises the following workflow:
#   1) Capturing new video frame.
#   2) Run face detection / tracking.
#   3) Extract face embedding and perform face identification (mode "ident") or re-identification
#      (mode "cluster").
#   4) Display face detection / tracking along with the prediction of face identification.


def main(args):
    # Setup OpenCV video capture.
    if args.video == "none":
        camera = cv2.VideoCapture(-1)
        wait_for_frame = 200
    else:
        camera = cv2.VideoCapture(args.video)
        wait_for_frame = 100
    camera.set(3, 640)
    camera.set(4, 480)

    # Image display
    cv2.namedWindow("Camera")
    cv2.moveWindow("Camera", 0, 0)

    # Prepare face detection, identification, and clustering.
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    clustering = FaceClustering()

    # The video capturing loop.
    on_track = False
    while True:
        key = cv2.waitKey(wait_for_frame)

        # Stop capturing using ESC.
        if (key & 255) == 27:
            break

        # Pause capturing using 'p'.
        if key == ord("p"):
            cv2.waitKey(-1)

        # Capture new video frame.
        _, frame = camera.read()
        if frame is None:
            print("End of stream")
            break
        # Resize the frame.
        height, width = frame.shape[:2]
        if width < 640:
            s = 640.0 / width
            frame = cv2.resize(frame, (int(s * width), int(s * height)))
        # Flip frame if it is live video.
        if args.video == "none":
            frame = cv2.flip(frame, 1)

        # Track (or initially detect if required) a face in the current frame.
        face = detector.track_face(frame)

        label_str = ""
        confidence_str = ""
        state_str = ""
        predicted_label = ""
        if face is not None and not on_track:
            # We found a new face that we can track over time.
            on_track = True

            if args.mode == ReIdMode.IDENT:
                # Face identification: predict identity for the current frame.
                predicted_label, prob, dist_to_prediction = recognizer.predict(face["aligned"])
                label_str = "{}".format(predicted_label)
                confidence_str = "Prob.: {:1.2f}, Dist.: {:1.2f}".format(prob, dist_to_prediction)
            elif args.mode == ReIdMode.CLUSTER:
                # Face clustering: determine cluster for the current frame.
                predicted_label, distances_to_clusters = clustering.predict(face["aligned"])
                label_str = "Cluster {}".format(predicted_label)
                confidence_str = "Dist.: {}".format(
                    np.array2string(distances_to_clusters, precision=2)
                )

            state_str = "{} | {}".format(label_str, confidence_str)

        if face is None or face["response"] < detector.tm_threshold:
            # We lost the track of the face visible in the previous frame.
            on_track = False

        # Display annotations for face tracking, identification, and clustering.
        if face is not None:
            face_rect = face["rect"]
            color = (0, 255, 0)
            if isinstance(predicted_label, str) and predicted_label.lower() == "unknown":
                color = (0, 0, 255)
            cv2.rectangle(
                frame,
                (face_rect[0], face_rect[1]),
                (face_rect[0] + face_rect[2] - 1, face_rect[1] + face_rect[3] - 1),
                color,
                2,
            )
            ((tw, th), _) = cv2.getTextSize(state_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (face_rect[0] - 1, face_rect[1] + face_rect[3]),
                (face_rect[0] + 1 + tw, face_rect[1] + face_rect[3] + th + 4),
                color,
                -1,
            )
            cv2.putText(
                frame,
                state_str,
                (face_rect[0], face_rect[1] + face_rect[3] + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        cv2.imshow("Camera", frame)


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=ReIdMode,
        choices=enum_choices(ReIdMode),
        default=ReIdMode.IDENT,
        help="The test mode.",
    )

    parser.add_argument(
        "--video",
        type=str,
        default=Config.TEST_DATA.joinpath("Alan_Ball", "%04d.jpg"),
        help="The video capture input. In case of 'none' the default video capture (webcam) is "
             "used. Use a filename(s) to read video data from image file (see VideoCapture "
             "documentation).",
    )

    return parser.parse_args()


def test51(source=0):
    # detector = FaceDetector(tm_window_size=25, tm_threshold=0.7)
    # result = detector.track_face(frame)
    # if result:
    #     x, y, w, h = result["rect"]
    #     score = result["response"]
    #     # draw box
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     # overlay match score
    #     cv2.putText(frame, f"{score:.2f}", (x, y - 5),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # cv2.imshow("Track", frame)
    # 1) Open video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source {source}")
        return

    # 2) Create your tracker
    tracker = FaceDetector(tm_window_size=25, tm_threshold=0.7)

    # 3) Read & process frames in a loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video or cannot fetch frame

        # 'frame' is a NumPy array (H×W×3) containing the current image
        result = tracker.track_face(frame)

        # If we got a detection/tracking result, draw it
        if result is not None:
            x, y, w, h = result["rect"]
            score = result["response"]
            # draw a green rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # overlay the match confidence
            cv2.putText(
                frame,
                f"{score:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        s = 224
        # show the frame
        cv2.imshow("Face Tracking", frame)

        # exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 4) Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(arguments())
    # test51(source=0)
