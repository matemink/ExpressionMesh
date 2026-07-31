from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

LANDMARK_COUNT = 468
COORDINATE_COUNT = 3
FEATURE_COUNT = LANDMARK_COUNT * COORDINATE_COUNT


def landmarks_to_features(
    landmarks: Iterable[Sequence[float]],
) -> NDArray[np.float32]:
    """Convert 468 (x, y, z) landmarks into the legacy model's 1404 features."""
    points = np.asarray(list(landmarks), dtype=np.float32)
    expected_shape = (LANDMARK_COUNT, COORDINATE_COUNT)
    if points.shape != expected_shape:
        raise ValueError(f"Expected landmark shape {expected_shape}, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Landmarks contain NaN or infinite values")

    # The committed model was trained with per-axis minimum subtraction.
    normalized = points - points.min(axis=0)
    return normalized.reshape(FEATURE_COUNT)


def validate_feature_vector(
    features: Sequence[float] | NDArray[np.floating],
) -> NDArray[np.float32]:
    vector = np.asarray(features, dtype=np.float32).reshape(-1)
    if vector.size != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} features, got {vector.size}")
    if not np.isfinite(vector).all():
        raise ValueError("Feature vector contains NaN or infinite values")
    return vector
