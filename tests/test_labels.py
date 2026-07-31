import pytest

from emotion_recognition.labels import Emotion, emotion_from_id


def test_stable_class_mapping() -> None:
    assert [emotion.name for emotion in Emotion] == ["HAPPY", "SAD", "SURPRISED"]
    assert emotion_from_id(2) is Emotion.SURPRISED


def test_unknown_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown emotion class"):
        emotion_from_id(3)
