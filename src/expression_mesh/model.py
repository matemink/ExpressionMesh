from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .features import FEATURE_COUNT, validate_feature_vector
from .labels import EXPECTED_CLASS_IDS, Expression, expression_from_id


class ModelContractError(ValueError):
    """Raised when a serialized estimator is incompatible with this pipeline."""


@dataclass(frozen=True)
class Prediction:
    expression: Expression
    confidence: float | None


class ExpressionClassifier:
    def __init__(self, estimator: Any) -> None:
        self._estimator = estimator
        self._validate_contract()

    @classmethod
    def load(cls, path: str | Path) -> ExpressionClassifier:
        """Load a trusted pickle artifact and validate it before inference."""
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with model_path.open("rb") as model_file:
            estimator = pickle.load(model_file)  # noqa: S301 - trusted local artifact only
        return cls(estimator)

    @property
    def estimator(self) -> Any:
        return self._estimator

    def predict(self, features: Any) -> Prediction:
        vector = validate_feature_vector(features)
        sample = vector.reshape(1, -1)
        class_id = self._estimator.predict(sample)[0]
        expression = expression_from_id(class_id)

        confidence = None
        if hasattr(self._estimator, "predict_proba"):
            probabilities = self._estimator.predict_proba(sample)[0]
            class_index = list(self._estimator.classes_).index(expression.value)
            confidence = float(probabilities[class_index])

        return Prediction(expression=expression, confidence=confidence)

    def _validate_contract(self) -> None:
        if not callable(getattr(self._estimator, "predict", None)):
            raise ModelContractError("Model must expose predict()")

        try:
            classes = tuple(
                expression_from_id(class_id).value
                for class_id in getattr(self._estimator, "classes_", ())
            )
        except (TypeError, ValueError) as error:
            raise ModelContractError(f"Invalid model classes: {error}") from error
        if classes != EXPECTED_CLASS_IDS:
            raise ModelContractError(
                f"Expected model classes {EXPECTED_CLASS_IDS}, got {classes or 'none'}"
            )

        feature_count = getattr(self._estimator, "n_features_in_", None)
        if feature_count != FEATURE_COUNT:
            raise ModelContractError(
                f"Expected a {FEATURE_COUNT}-feature model, got {feature_count!r}"
            )
