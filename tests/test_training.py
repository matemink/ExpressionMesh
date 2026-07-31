import numpy as np
import pytest

from emotion_recognition.features import FEATURE_COUNT
from emotion_recognition.training import load_dataset, train_classifier


def test_training_is_seeded_and_returns_complete_confusion_matrix() -> None:
    generator = np.random.default_rng(7)
    features = generator.normal(size=(60, FEATURE_COUNT))
    labels = np.repeat(np.arange(3), 20)

    classifier, metrics = train_classifier(
        features,
        labels,
        random_state=123,
        n_estimators=5,
        test_size=0.2,
    )

    assert classifier.random_state == 123
    assert metrics.random_state == 123
    assert metrics.training_samples == 48
    assert metrics.test_samples == 12
    assert np.asarray(metrics.confusion_matrix).shape == (3, 3)

    second_classifier, second_metrics = train_classifier(
        features,
        labels,
        random_state=123,
        n_estimators=5,
        test_size=0.2,
    )
    assert np.array_equal(classifier.predict(features), second_classifier.predict(features))
    assert metrics == second_metrics


def test_dataset_rejects_fractional_labels(tmp_path) -> None:
    rows = np.zeros((3, FEATURE_COUNT + 1))
    rows[:, -1] = [0, 1, 1.5]
    dataset_path = tmp_path / "data.txt"
    np.savetxt(dataset_path, rows)

    with pytest.raises(ValueError, match="labels must be integers"):
        load_dataset(dataset_path)


def test_dataset_requires_every_model_class(tmp_path) -> None:
    rows = np.zeros((2, FEATURE_COUNT + 1))
    rows[:, -1] = [0, 1]
    dataset_path = tmp_path / "data.txt"
    np.savetxt(dataset_path, rows)

    with pytest.raises(ValueError, match="Expected classes"):
        load_dataset(dataset_path)
