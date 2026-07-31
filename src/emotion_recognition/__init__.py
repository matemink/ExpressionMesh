"""Face emotion recognition pipeline."""

from .labels import Emotion
from .model import EmotionClassifier, Prediction

__all__ = ["Emotion", "EmotionClassifier", "Prediction"]
__version__ = "0.2.0"
