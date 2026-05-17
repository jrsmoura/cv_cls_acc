from typing import Tuple
import torch

IMG_HEIGHT: int = 224
IMG_WIDTH: int = 224
IMG_SIZE: Tuple[int, int] = (224, 224)

EPOCHS: int = 10
BATCH_SIZE: int = 64
VALIDATION_SPLIT: float = 0.20
LEARNING_RATE: float = 0.0001
MOMENTUM: float = 0.9

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
