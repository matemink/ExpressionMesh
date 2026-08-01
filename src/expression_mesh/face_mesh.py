from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from .features import landmarks_to_features


class FaceLandmarkExtractor:
    """Owns one MediaPipe FaceMesh instance for an entire image or video session."""

    def __init__(
        self,
        *,
        static_image_mode: bool,
        min_detection_confidence: float = 0.5,
    ) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                'Vision dependencies are missing. Install with: pip install -e ".[vision]"'
            ) from error

        self._cv2 = cv2
        self._mp = mp
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
        )

    def extract(self, image: Any, *, draw: bool = False) -> NDArray | None:
        if image is None:
            raise ValueError("Cannot extract landmarks from an empty image")

        rgb_image = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        if draw:
            drawing = self._mp.solutions.drawing_utils
            drawing.draw_landmarks(
                image=image,
                landmark_list=face,
                connections=self._mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing.DrawingSpec(thickness=1, circle_radius=1),
                connection_drawing_spec=drawing.DrawingSpec(thickness=1, circle_radius=1),
            )

        return landmarks_to_features((point.x, point.y, point.z) for point in face.landmark)

    def close(self) -> None:
        self._face_mesh.close()

    def __enter__(self) -> FaceLandmarkExtractor:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
