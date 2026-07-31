from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from .model import EmotionClassifier


def describe_model(model_path: Path) -> dict[str, object]:
    classifier = EmotionClassifier.load(model_path)
    estimator = classifier.estimator
    return {
        "artifact": str(model_path),
        "estimator": f"{type(estimator).__module__}.{type(estimator).__name__}",
        "classes": [int(class_id) for class_id in estimator.classes_],
        "feature_count": int(estimator.n_features_in_),
        "estimators": int(estimator.n_estimators),
        "random_state": estimator.random_state,
        "scikit_learn_runtime": version("scikit-learn"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and describe a trusted model artifact")
    parser.add_argument("model", type=Path, nargs="?", default=Path("model"))
    args = parser.parse_args()
    print(json.dumps(describe_model(args.model), indent=2))


if __name__ == "__main__":
    main()
