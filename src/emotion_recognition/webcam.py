from __future__ import annotations

import argparse
from pathlib import Path

from .face_mesh import FaceLandmarkExtractor
from .model import EmotionClassifier


def run_webcam(model_path: Path, camera_index: int) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError('OpenCV is missing. Install with: pip install -e ".[vision]"') from error

    classifier = EmotionClassifier.load(model_path)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    try:
        with FaceLandmarkExtractor(static_image_mode=False) as extractor:
            while True:
                received, frame = capture.read()
                if not received:
                    raise RuntimeError("Camera stopped returning frames")

                features = extractor.extract(frame, draw=True)
                label = "NO FACE"
                if features is not None:
                    prediction = classifier.predict(features)
                    confidence = (
                        f" {prediction.confidence:.0%}" if prediction.confidence is not None else ""
                    )
                    label = f"{prediction.emotion.name}{confidence}"

                cv2.putText(
                    frame,
                    label,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (40, 220, 40),
                    3,
                )
                cv2.imshow("Emotion recognition | q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-time emotion recognition")
    parser.add_argument("--model", type=Path, default=Path("model"))
    parser.add_argument("--camera", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_webcam(args.model, args.camera)


if __name__ == "__main__":
    main()
