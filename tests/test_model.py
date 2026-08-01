import numpy as np
import pytest

from expression_mesh.features import FEATURE_COUNT
from expression_mesh.labels import Expression
from expression_mesh.model import ExpressionClassifier, ModelContractError


class FakeEstimator:
    classes_ = np.asarray([0, 1, 2])
    n_features_in_ = FEATURE_COUNT

    def predict(self, _sample: np.ndarray) -> np.ndarray:
        return np.asarray([1])

    def predict_proba(self, _sample: np.ndarray) -> np.ndarray:
        return np.asarray([[0.1, 0.8, 0.1]])


def test_prediction_maps_class_and_confidence() -> None:
    prediction = ExpressionClassifier(FakeEstimator()).predict(np.zeros(FEATURE_COUNT))

    assert prediction.expression is Expression.SAD
    assert prediction.confidence == pytest.approx(0.8)


def test_model_with_missing_class_is_rejected() -> None:
    estimator = FakeEstimator()
    estimator.classes_ = np.asarray([0, 1])

    with pytest.raises(ModelContractError, match="Expected model classes"):
        ExpressionClassifier(estimator)
