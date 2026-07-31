from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .face_mesh import FaceLandmarkExtractor
from .labels import EMOTIONS

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class DatasetReport:
    accepted: int
    no_face: int
    unreadable: int


def prepare_dataset(source: Path, output: Path) -> DatasetReport:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError('OpenCV is missing. Install with: pip install -e ".[vision]"') from error

    records: list[np.ndarray] = []
    no_face = 0
    unreadable = 0

    with FaceLandmarkExtractor(static_image_mode=True) as extractor:
        for emotion in EMOTIONS:
            class_directory = source / emotion.directory_name
            if not class_directory.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_directory}")

            accepted_for_class = 0
            image_paths = sorted(
                path
                for path in class_directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for image_path in image_paths:
                image = cv2.imread(str(image_path))
                if image is None:
                    unreadable += 1
                    continue

                features = extractor.extract(image)
                if features is None:
                    no_face += 1
                    continue
                records.append(np.append(features, emotion.value))
                accepted_for_class += 1

            if accepted_for_class == 0:
                raise RuntimeError(f"No usable {emotion.directory_name} faces in {class_directory}")

    if not records:
        raise RuntimeError(f"No usable faces found under {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, np.asarray(records, dtype=np.float32))
    return DatasetReport(accepted=len(records), no_face=no_face, unreadable=unreadable)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks into a dataset")
    parser.add_argument("source", type=Path, help="Directory containing happy/sad/surprised")
    parser.add_argument("--output", type=Path, default=Path("data.txt"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = prepare_dataset(args.source, args.output)
    print(f"Dataset written to {args.output}")
    print(
        f"Accepted: {report.accepted} | No face: {report.no_face} | Unreadable: {report.unreadable}"
    )


if __name__ == "__main__":
    main()
