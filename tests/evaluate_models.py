import os.path

import torch
from typing import Tuple
import config


def eval_model(model_path: str) -> Tuple[float, float]:
    pass


if __name__ == '__main__':
    model_name: str = "vgg16_regression.pth"
    model_path = os.path.join(config.MODELS_DIR, model_name)
    eval_model(model_path)
