# ExpressionMesh

[![CI](https://github.com/matemink/ExpressionMesh/actions/workflows/ci.yml/badge.svg)](https://github.com/matemink/ExpressionMesh/actions/workflows/ci.yml)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-3776AB.svg)](https://www.python.org/)

ExpressionMesh is an end-to-end synthetic-data-to-webcam computer-vision project. It covers
the complete path:

- generate a labeled portrait dataset with Stable Diffusion v1.5 and Hugging Face Diffusers;
- extract 468 facial landmarks per image with MediaPipe Face Mesh;
- train and evaluate a deterministic scikit-learn Random Forest classifier;
- run local real-time inference from a webcam with OpenCV and MediaPipe.

This is an engineering experiment, not a claim that facial expressions reveal a person's
internal emotional state. See the [model card](MODEL_CARD.md) for provenance and limitations.

## Pipeline

```mermaid
flowchart LR
    subgraph Training["Dataset generation and model training"]
        Prompts["Expression prompts"] --> SD["Stable Diffusion v1.5"]
        SD --> Dataset["Labeled synthetic portraits"]
        Dataset --> TrainMesh["MediaPipe Face Mesh"]
        TrainMesh --> TrainFeatures["1,404 normalized landmarks"]
        TrainFeatures --> Train["Random Forest training"]
    end

    subgraph Runtime["Real-time webcam inference"]
        Webcam["Webcam"] --> OpenCV["OpenCV frame capture"]
        OpenCV --> LiveMesh["MediaPipe Face Mesh"]
        LiveMesh --> LiveFeatures["1,404 normalized landmarks"]
    end

    Train --> Model["Trained model"]
    LiveFeatures --> Model
    Model --> Result["HAPPY | SAD | SURPRISED<br/>with confidence"]
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
expression-mesh-model-info model
```

Start webcam inference and press `q` to exit:

```bash
expression-mesh-webcam --model model --camera 0
```

Only load pickle artifacts you trust. Loading a malicious pickle can execute arbitrary code.

## Reproduce the pipeline

The input directory must contain `happy`, `sad`, and `surprised` subdirectories.

```bash
expression-mesh-prepare-data path/to/faces --output data.txt
expression-mesh-train data.txt --model artifacts/model.pkl --metrics artifacts/metrics.json
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
| `src/expression_mesh/features.py` | Landmark normalization and feature validation |
| `src/expression_mesh/face_mesh.py` | Reusable MediaPipe adapter |
| `src/expression_mesh/model.py` | Serialized model contract and predictions |
| `src/expression_mesh/dataset.py` | Image-to-feature dataset preparation |
| `src/expression_mesh/training.py` | Deterministic train/evaluate/save workflow |
| `src/expression_mesh/webcam.py` | Real-time OpenCV application |
| `tests/` | Fast hardware-independent tests |
| `tools/` | Optional synthetic generation and CUDA diagnostics |

## Current limitations

- The committed legacy model was trained on synthetic images only.
- Its original dataset and evaluation metrics are unavailable, so no accuracy is claimed.
- The existing artifact requires scikit-learn 1.4.1.post1 for reliable pickle compatibility.
- Real-world validation across cameras, lighting, poses, and demographics is future work.
