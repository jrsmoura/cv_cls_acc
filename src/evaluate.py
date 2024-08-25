import argparse
from typing import Dict, Tuple, List

import torch
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix)

from models.models_pt import (
    VGG16RegressionModel,
    ResNet50RegressionModel,
    EfficientNetRegressionModel,
    MobileNetV3LRegressionModel
)
from auxiliary_functions.aux_functions import (
    get_filepaths_and_labels,
    create_dataloader
)


def evaluate_model(model: torch, val_loader: torch) -> Tuple[List, List]:
    """

    :param model:
    :param val_loader:
    :return: (labels, preds): Tuple[List, List]
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    labels = []
    preds = []

    with torch.no_grad():
        for batch in val_loader:
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                labels.extend(labels.cpu().numpy())
                preds.extend(predicted.cpu().numpy())

    return labels, preds


def calculate_metrics(labels, preds) -> Dict:
    """

    :param labels:
    :param preds:
    :return: metrics: Dict
    """
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    cm = confusion_matrix(labels, preds)

    metrics: Dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist()
    }


def evaluate(model_path: str, data_dir: str, metrics_file):
    """
    Evaluate the model on the test set
    :param model_path:
    :param data_dir:
    :param metrics_file:
    :return:
    """
    file_paths, labels = get_filepaths_and_labels(data_dir)
    val_loader: object
    _, val_loader, _, _ = create_dataloader(file_paths, labels, batch_size=32)

    model = None
    if "VGG16" in model_path:
        model = VGG16RegressionModel()
    elif "ResNet50" in model_path:
        model = ResNet50RegressionModel()
    elif "EfficientNet" in model_path:
        model = EfficientNetRegressionModel()
    elif "MobileNetV3L" in model_path:
        model = MobileNetV3LRegressionModel()

    model.load_state_dict(torch.load(model_path))
    metrics = evaluate_model(model, val_loader)

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--metrics_file', type=str, required=True)

    args = parser.parse_args()
    evaluate(args.model_path, args.data_dir, args.metrics_file)
