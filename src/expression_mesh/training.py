from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from .features import FEATURE_COUNT
from .labels import EXPECTED_CLASS_IDS


@dataclass(frozen=True)
class TrainingMetrics:
    accuracy: float
    confusion_matrix: list[list[int]]
    training_samples: int
    test_samples: int
    random_state: int


def load_dataset(path: Path) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rows = np.atleast_2d(np.loadtxt(path))
    expected_columns = FEATURE_COUNT + 1
    if rows.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} columns, got {rows.shape[1]}")
    if not np.isfinite(rows).all():
        raise ValueError("Dataset contains NaN or infinite values")

    features = rows[:, :-1]
    raw_labels = rows[:, -1]
    if not np.equal(raw_labels, np.rint(raw_labels)).all():
        raise ValueError("Dataset labels must be integers")
    labels = raw_labels.astype(np.int64)
    classes = tuple(int(class_id) for class_id in np.unique(labels))
    if classes != EXPECTED_CLASS_IDS:
        raise ValueError(f"Expected classes {EXPECTED_CLASS_IDS}, got {classes}")
    return features, labels


def train_classifier(
    features: NDArray,
    labels: NDArray,
    *,
    random_state: int = 42,
    n_estimators: int = 300,
    test_size: float = 0.2,
) -> tuple[RandomForestClassifier, TrainingMetrics]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=labels,
    )
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    metrics = TrainingMetrics(
        accuracy=float(accuracy_score(y_test, predictions)),
        confusion_matrix=confusion_matrix(y_test, predictions, labels=EXPECTED_CLASS_IDS).tolist(),
        training_samples=len(x_train),
        test_samples=len(x_test),
        random_state=random_state,
    )
    return classifier, metrics


def save_training_run(
    classifier: RandomForestClassifier,
    metrics: TrainingMetrics,
    *,
    model_path: Path,
    metrics_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(classifier, model_file)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the facial-expression classifier")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, default=Path("model"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--estimators", type=int, default=300)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    features, labels = load_dataset(args.dataset)
    classifier, metrics = train_classifier(
        features,
        labels,
        random_state=args.random_state,
        n_estimators=args.estimators,
    )
    save_training_run(
        classifier,
        metrics,
        model_path=args.model,
        metrics_path=args.metrics,
    )
    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(f"Model: {args.model}")
    print(f"Metrics: {args.metrics}")


if __name__ == "__main__":
    main()
