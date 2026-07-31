from enum import IntEnum


class Emotion(IntEnum):
    """Stable class identifiers used by both datasets and serialized models."""

    HAPPY = 0
    SAD = 1
    SURPRISED = 2

    @property
    def directory_name(self) -> str:
        return self.name.lower()


EMOTIONS = tuple(Emotion)
EXPECTED_CLASS_IDS = tuple(emotion.value for emotion in EMOTIONS)


def emotion_from_id(class_id: int | float) -> Emotion:
    numeric_id = int(class_id)
    if numeric_id != class_id:
        raise ValueError(f"Emotion class must be an integer, got {class_id!r}")

    try:
        return Emotion(numeric_id)
    except ValueError as error:
        raise ValueError(f"Unknown emotion class: {numeric_id}") from error
