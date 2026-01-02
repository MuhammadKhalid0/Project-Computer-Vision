import argparse

import cv2

from cvproj_exc.config import Config, ReIdMode, enum_choices
from cvproj_exc.face_detector import FaceDetector
from cvproj_exc.face_recognition import FaceClustering, FaceRecognizer

# The training module of the face recognition system. In summary, training comprises the following
# workflow:
#   1) Capturing new video frame.
#   2) Run face detection / tracking.
#   3) Extract face embedding and update face identification (mode "ident") or clustering
#      (mode "cluster").
#   4) Fit face identification (mode "ident") or clustering (mode "cluster") and save trained
#      models.


def main(args):
    # Setup OpenCV video capture.
    if args.video == "none":
        camera = cv2.VideoCapture(-1)
        wait_for_frame = 1
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
    clustering = FaceClustering(num_clusters=6)

    # The video capturing loop.
    state = ""
    num_samples = 0
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
        if (face := detector.track_face(frame)) is not None:
            # We detected a face in the current frame.
            num_samples = num_samples + 1

            if args.mode == ReIdMode.IDENT:
                # Update face identification.
                recognizer.partial_fit(face.aligned, args.label)
                state = "{} ({} samples)".format(args.label, num_samples)
            elif args.mode == ReIdMode.CLUSTER:
                # Update face clustering.
                clustering.partial_fit(face.aligned)
                state = "{} samples".format(num_samples)

            # Display annotations for face tracking and training.
            face_rect = face.rect
            cv2.rectangle(
                frame,
                (face_rect[0], face_rect[1]),
                (face_rect[0] + face_rect[2] - 1, face_rect[1] + face_rect[3] - 1),
                (0, 255, 0),
                2,
            )
            ((tw, th), _) = cv2.getTextSize(state, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (face_rect[0] - 1, face_rect[1] + face_rect[3]),
                (face_rect[0] + 1 + tw, face_rect[1] + face_rect[3] + th + 4),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                frame,
                state,
                (face_rect[0], face_rect[1] + face_rect[3] + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        cv2.imshow("Camera", frame)

    # Save trained models for face identification and clustering.
    if args.mode == ReIdMode.IDENT:
        print("Save trained face recognition model")
        recognizer.save()

    if args.mode == ReIdMode.CLUSTER:
        # Incremental k-means: fit clustering after collecting embeddings.
        # Uses existing cluster centers as initialization if available, allowing smooth adaptation
        # to new data while preserving previous cluster structure.
        total_embeddings = len(clustering.embeddings)
        
        if total_embeddings >= clustering.num_clusters:
            # Only fit if we have at least as many embeddings as clusters
            print(f"Fitting clustering on {total_embeddings} total embeddings...")
            clustering.fit()
        else:
            print(f"Collected {total_embeddings} embeddings (need at least {clustering.num_clusters} for clustering). Skipping fit.")
        
        print("Save trained face clustering")
        clustering.save()


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=ReIdMode,
        choices=enum_choices(ReIdMode),
        default=ReIdMode.IDENT,
        help="The training mode.",
    )

    parser.add_argument(
        "--video",
        type=str,
        default=Config.TRAIN_DATA.joinpath("Alan_Ball", "%04d.jpg"),
        help="The video capture input. In case of 'none' the default video capture (webcam) is "
        "used. Use a filename(s) to read video data from image file (see VideoCapture "
        "documentation).",
    )

    parser.add_argument(
        "--label",
        type=str,
        default="Alan_Ball",
        help="Identity label (only required for face identification).",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    main(arguments())
