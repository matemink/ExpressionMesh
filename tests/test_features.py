import numpy as np
import pytest

from expression_mesh.features import FEATURE_COUNT, landmarks_to_features


def test_landmarks_to_features_preserves_legacy_normalization() -> None:
    points = np.column_stack(
        (
            np.arange(468, dtype=np.float32) + 10,
            np.arange(468, dtype=np.float32) * 2 + 20,
            np.arange(468, dtype=np.float32) * -1 - 30,
        )
    )

    features = landmarks_to_features(points)

    assert features.shape == (FEATURE_COUNT,)
    assert features[:3] == pytest.approx([0.0, 0.0, 467.0])
    assert features[-3:] == pytest.approx([467.0, 934.0, 0.0])


def test_landmarks_to_features_rejects_incomplete_face() -> None:
    with pytest.raises(ValueError, match="Expected landmark shape"):
        landmarks_to_features([(0.0, 0.0, 0.0)] * 467)
