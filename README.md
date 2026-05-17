# CV Classification Accelerator

A PyTorch-based transfer learning framework for image regression tasks. The project uses pre-trained ImageNet backbones with frozen weights and attaches a custom regression head, targeting continuous label prediction (e.g., age estimation from face images).

---

## Table of Contents

- [CV Classification Accelerator](#cv-classification-accelerator)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Models](#models)
  - [Data Pipeline](#data-pipeline)
    - [Directory Structure Expected](#directory-structure-expected)
    - [Key Functions — `auxiliary_functions/aux_functions.py`](#key-functions--auxiliary_functionsaux_functionspy)
  - [Configuration](#configuration)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Quick start — `main.py`](#quick-start--mainpy)
    - [CLI training — `src/train.py`](#cli-training--srctrainpy)
    - [Training loop behavior](#training-loop-behavior)
    - [Upload trained model to GCS](#upload-trained-model-to-gcs)
  - [Finetuning](#finetuning)
  - [Evaluation](#evaluation)
  - [MLOps \& Pipeline Orchestration](#mlops--pipeline-orchestration)
    - [DVC — `dvc.yaml`](#dvc--dvcyaml)
    - [Dataset builder — `src/build_datasets.py`](#dataset-builder--srcbuild_datasetspy)
    - [Jenkins — `mlops/Jenkinsfile`](#jenkins--mlopsjenkinsfile)
  - [Tests](#tests)
  - [Requirements](#requirements)
  - [TO DO](#to-do)
  - [Author](#author)

---

## Overview

This project provides a modular, plug-and-play framework to benchmark and train multiple computer vision regression models using transfer learning. All models share the same frozen-backbone + custom-head pattern, making it easy to swap architectures and compare results.

**Primary use case:** Age regression from facial images, where subdirectory names represent the numeric label (age).

---

## Project Structure

```
cv_cls_acc/
├── accelerator/
│   └── accelerator.py            # (placeholder)
├── auxiliary_functions/
│   ├── __init__.py
│   └── aux_functions.py          # Dataset loading, preprocessing, DataLoader creation
├── finetuning/
│   └── finetuning.py             # Grid-search hyperparameter tuning
├── mlops/
│   └── Jenkinsfile               # CI/CD pipeline definition (placeholder)
├── models/
│   ├── __init__.py
│   └── models_pt.py              # All regression model architectures
├── src/
│   ├── build_datasets.py         # Utility to sample and build sub-datasets
│   ├── evaluate.py               # CLI evaluation script
│   ├── preprocess.py             # CLI preprocessing script
│   └── train.py                  # CLI training script
├── tests/
│   └── test_models_pt.py         # Unit tests for model architectures
├── config.py                     # Global hyperparameters and paths
├── dvc.yaml                      # DVC pipeline: preprocess → train → evaluate
├── main.py                       # Main training entry point
├── Makefile                      # Dev automation (setup, data download, run, upload)
├── params.yaml                   # Per-model DVC parameters
├── pyproject.toml                # Package metadata (uv/PEP 517)
├── requirements.txt              # Pinned dependencies
└── setup.py                      # Setuptools package config
```

---

## Models

All models are defined in `models/models_pt.py` and follow the same architecture pattern:

```
Pre-trained Backbone (frozen)
        ↓
Global Average Pooling (AdaptiveAvgPool2d → 1×1)
        ↓
Flatten
        ↓
Linear → ReLU  (custom head)
        ↓
Linear → scalar output  (regression)
```

| Model Class | Backbone | Backbone Output Channels | Head Input |
|---|---|---|---|
| `VGG16RegressionModel` | VGG16 (features only) | 512 | 512 → 256 → 1 |
| `ResNet50RegressionModel` | ResNet50 (no classifier) | 2048 | 2048 → 256 → 1 |
| `EfficientNetRegressionModel` | EfficientNet-B0 (no classifier) | 1280 | 1280 → 256 → 1 |
| `InceptionV3RegressionModel` | InceptionV3 (no classifier) | 2048 | 2048 → 256 → 1 |
| `MobileNetV3LRegressionModel` | MobileNetV3-Large (no classifier) | 1280 | 1280 → 256 → 1 |

All backbone weights are loaded from ImageNet pre-trained defaults and **frozen** (`requires_grad=False`). Only the regression head is trained.

---

## Data Pipeline

### Directory Structure Expected

```
data/
├── 18/
│   ├── image1.jpg
│   └── image2.jpg
├── 25/
│   └── image3.jpg
...
```

Each subdirectory name is the numeric label (e.g., age). The pipeline reads them as `float`.

### Key Functions — `auxiliary_functions/aux_functions.py`

| Function | Description |
|---|---|
| `get_filepaths_and_labels(data_dir)` | Walks the labeled subdirectory structure and returns parallel lists of file paths and float labels |
| `process_data_pt(image_path, label)` | Loads a single image, normalizes to `[0, 1]`, resizes to `IMG_SIZE` |
| `create_dataloader(file_paths, labels, batch_size, val_split=0.2)` | Builds `CustomImageDataset`, splits into train/val, returns two `DataLoader`s plus sizes |
| `save_model(epochs, model, optimizer, criterion, pretrained)` | Persists full training checkpoint to disk |
| `CustomImageDataset` | PyTorch `Dataset` subclass; handles grayscale→RGB expansion, applies transforms |

**Image normalization** uses ImageNet statistics:
- mean: `[0.485, 0.456, 0.406]`
- std:  `[0.229, 0.224, 0.225]`

---

## Configuration

All global parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `IMG_HEIGHT` / `IMG_WIDTH` | 224 | Input image dimensions |
| `IMG_SIZE` | `(224, 224)` | Tuple used by transforms |
| `EPOCHS` | 10 | Training epochs |
| `BATCH_SIZE` | 64 | Samples per batch |
| `VALIDATION_SPLIT` | 0.20 | Fraction reserved for validation |
| `LEARNING_RATE` | 0.0001 | Adam learning rate |
| `MOMENTUM` | 0.9 | Momentum (for SGD/RMSProp) |
| `DATA_DIR` | `data/` | Root data directory |
| `MODELS_DIR` | `models/` | Model output directory |
| `MODEL_ID` | `google/vit-base-patch16-224-in21k` | HuggingFace ViT identifier |
| `DEVICE` | `cuda` / `cpu` | Auto-detected via `torch.cuda.is_available()` |

Per-model DVC parameters are in `params.yaml` (epochs, batch_size, learning_rate, optimizer, loss per architecture).

---

## Installation

```bash
# Create virtual environment and install dependencies
make setup

# Or manually with uv
uv sync

# Or with pip
pip install -r requirements.txt
pip install -e .
```

**Download training data from GCS:**

```bash
make data
# equivalent to:
# gsutil -m cp -r gs://verifymy-ai-trainning/research-dataset/images/age-regression .
```

---

## Usage

### Quick start — `main.py`

Trains with `VGG16RegressionModel` by default. Edit the commented lines to switch models.

```bash
python main.py
```

### CLI training — `src/train.py`

```bash
python src/train.py \
  --model_name VGG16RegressionModel \
  --data_dir data/ \
  --output_model_path saved_models/vgg16.pth \
  --batch_size 32
```

Available `--model_name` values:
- `VGG16RegressionModel`
- `ResNet50RegressionModel`
- `EfficientNetRegressionModel`
- `MobileNetV3LRegressionModel`

### Training loop behavior

- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`
- **Early stopping**: patience of 3 epochs on validation loss
- Best model checkpoint saved automatically

### Upload trained model to GCS

```bash
make upload
# gsutil cp vgg_model.pth gs://verifymy-ai-trainning/
```

---

## Finetuning

`finetuning/finetuning.py` implements a **grid search** over:

| Hyperparameter | Values |
|---|---|
| Learning rate | Log-spaced from `1e-5` to `1e-1` (5 values) |
| Momentum | `[0.9, 0.95, 0.99]` |
| Optimizer | `SGD`, `Adam`, `RMSProp`, `AdamW` |

For each combination it trains for `num_epochs`, evaluates validation loss, and saves the best checkpoint as `best_model.pth`. Returns the best parameter dictionary.

```python
from finetuning.finetuning import finetune_model
best_params = finetune_model(model, train_loader, criterion, num_epochs=5,
                             learning_rates=..., momentums=..., optimizers=...)
```

---

## Evaluation

`src/evaluate.py` loads a saved model, runs inference on the validation split, and writes a JSON metrics file.

```bash
python src/evaluate.py \
  --model_path saved_models/vgg16.pth \
  --data_dir data/ \
  --metrics_file metrics/vgg16.json
```

**Computed metrics** (via scikit-learn):
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 score (weighted)
- Confusion matrix

The model architecture is inferred automatically from the model path filename (`VGG16`, `ResNet50`, `EfficientNet`, `MobileNetV3L`).

---

## MLOps & Pipeline Orchestration

### DVC — `dvc.yaml`

Defines a reproducible ML pipeline with three parameterized stages:

```
preprocess (foreach ds1, ds2, ds3, ds4)
    ↓
train (foreach VGG16, ResNet50, EfficientNet, MobileNetV3L)
    ↓
evaluate → metrics/<model>_<dataset>.json
```

Run the full pipeline:

```bash
dvc repro
```

Track experiments and compare metrics:

```bash
dvc metrics show
dvc metrics diff
```

### Dataset builder — `src/build_datasets.py`

Utility to sample a subset of the full dataset:

```bash
# Copies 10 random images from folders[5:10] in data/ into datasets/ds2/cls1/
python src/build_datasets.py
```

### Jenkins — `mlops/Jenkinsfile`

CI/CD pipeline definition (currently a placeholder for future automation).

---

## Tests

Unit tests for model architectures are in `tests/test_models_pt.py`.

```bash
pytest tests/
```

**Test coverage for `VGG16RegressionModel`:**

| Test | What it verifies |
|---|---|
| `test_instantiation` | Model constructs without errors |
| `test_forward_output_shape` | Forward pass returns shape `(batch, 1)` for regression |
| `test_backbone_is_frozen` | All backbone parameters have `requires_grad=False` |
| `test_head_is_trainable` | Dense and output layer parameters are trainable |

---

## Requirements

**Core runtime** (`pyproject.toml`):

| Package | Version |
|---|---|
| Python | >=3.11 |
| PyTorch | >=2.12.0 |
| torchvision | >=0.27.0 |
| torchaudio | >=2.11.0 |
| TensorFlow | >=2.21.0 |
| tqdm | >=4.67.3 |

**Additional** (`requirements.txt`): scikit-learn, numpy, pandas, matplotlib, Pillow, transformers, tensorboard, scipy, and more.

GPU support requires CUDA 12.1 (`torch==2.3.1+cu121` in dev requirements).

---

## TO DO

- [X] Review models implementation
- [X] CLI implementation
- [X] Benchmark dataset for tests
- [X] Setup as lib
- [ ] Implement more models
- [X] Different versions of VGG
- [ ] Different versions of ResNet
- [X] Insert a fine-tuning option
- [ ] Define a orchestrator

---

## Author

**Roberto Steiner** — jr.steiner@outlook.com

License: MIT
