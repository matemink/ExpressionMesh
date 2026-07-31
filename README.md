# Face Emotion Recognition

[![CI](https://github.com/matemink/emotion-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/matemink/emotion-recognition/actions/workflows/ci.yml)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-3776AB.svg)](https://www.python.org/)

A small, reproducible computer-vision pipeline that turns MediaPipe facial landmarks into
three expression classes using a scikit-learn Random Forest. The repository includes dataset
preparation, deterministic training, model validation, and a real-time webcam demo.

This is an engineering experiment, not a claim that facial expressions reveal a person's
internal emotional state. See the [model card](MODEL_CARD.md) for provenance and limitations.

## Pipeline

```text
synthetic portraits
       |
       v
MediaPipe Face Mesh (468 x/y/z landmarks)
       |
       v
minimum-per-axis normalization (1,404 features)
       |
       v
Random Forest classifier
       |
       v
HAPPY | SAD | SURPRISED + confidence
```

## Quick start

Create a Python 3.10-3.12 virtual environment, then install the webcam dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[vision]"
```

Validate the committed model contract:

```bash
emotion-model-info model
```

Start webcam inference and press `q` to exit:

```bash
emotion-webcam --model model --camera 0
```

Only load pickle artifacts you trust. Loading a malicious pickle can execute arbitrary code.

## Reproduce the pipeline

The input directory must contain `happy`, `sad`, and `surprised` subdirectories.

```bash
emotion-prepare-data path/to/faces --output data.txt
emotion-train data.txt --model artifacts/model.pkl --metrics artifacts/metrics.json
```

Training is seeded and emits both the model and machine-readable evaluation metadata. To
regenerate synthetic portraits, install `.[generation]` and run:

```bash
python tools/generate_synthetic_faces.py path/to/faces --per-class 250 --seed 42
```

The generated dataset is intentionally not tracked because of its size and external model
licensing. Preserve its generation parameters and inspect samples before training.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

GitHub Actions runs linting and unit tests on Python 3.10 and 3.12. Unit tests cover the stable
label mapping, the 1,404-feature transformation, incompatible model rejection, confidence
mapping, and deterministic training configuration. Hardware-dependent webcam behavior remains
an explicit manual test.

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/emotion_recognition/features.py` | Landmark normalization and feature validation |
| `src/emotion_recognition/face_mesh.py` | Reusable MediaPipe adapter |
| `src/emotion_recognition/model.py` | Serialized model contract and predictions |
| `src/emotion_recognition/dataset.py` | Image-to-feature dataset preparation |
| `src/emotion_recognition/training.py` | Deterministic train/evaluate/save workflow |
| `src/emotion_recognition/webcam.py` | Real-time OpenCV application |
| `tests/` | Fast hardware-independent tests |
| `tools/` | Optional synthetic generation and CUDA diagnostics |

## Current limitations

- The committed legacy model was trained on synthetic images only.
- Its original dataset and evaluation metrics are unavailable, so no accuracy is claimed.
- The existing artifact requires scikit-learn 1.4.1.post1 for reliable pickle compatibility.
- Real-world validation across cameras, lighting, poses, and demographics is future work.
