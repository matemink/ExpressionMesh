"""ExpressionMesh facial-expression classification pipeline."""

from .labels import Expression
from .model import ExpressionClassifier, Prediction

__all__ = ["Expression", "ExpressionClassifier", "Prediction"]
__version__ = "0.2.0"
