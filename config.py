from typing import Tuple
import torch

# image parameters
IMG_HEIGHT: int = 224
IMG_WIDTH: int = 224
IMG_SIZE: Tuple[int, int] = (224, 224)
# model parameters
EPOCHS: int = 10
BATCH_SIZE: int = 64
VALIDATION_SPLIT: float = 0.20
LEARNING_RATE: float = 0.0001
MOMENTUM: float = 0.9

# local parameters
DATA_DIR: str = 'data/'
MODELS_DIR: str = 'models/'
# ViT parameters
MODEL_ID: str = 'google/vit-base-patch16-224-in21k'

# Pytorch parameters
DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
