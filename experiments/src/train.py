import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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


def train_model(model, train_loader, val_loader, epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), learning_rate)
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    print("Starting:")
    print("================================================")

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)

                val_loss += loss.item() * images.size(0)

        val_loss /= len(train_loader.dataset)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Check for improvement in validation loss and implement early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       'saved_models/mobileNetV3Large_model_v1.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break


def train(model_name: str,
          data_dir: str,
          batch_size: int,
          output_model_path: str) -> None:
    """
    Main training script for the regression models.

    This script initializes the dataset, creates dataloaders for training and validation,
    and trains the chosen regression model (VGG16 or ResNet50). It uses Mean Squared Error (MSE)
    as the loss function and Adam as the optimizer. Training stops early if validation loss
    does not improve for a given patience period.
    """
    # Retrieve file paths and labels for the dataset
    global model
    file_paths, labels = get_filepaths_and_labels(data_dir)

    # Create dataloaders for training and validation
    train_loader, val_loader, train_size, val_size = create_dataloader(
        file_paths,
        labels,
        batch_size
    )

    # Selecionar o modelo
    if model_name == "VGG16RegressionModel":
        model = VGG16RegressionModel()
    elif model_name == "ResNet50RegressionModel":
        model = ResNet50RegressionModel()
    elif model_name == "EfficientNetRegressionModel":
        model = EfficientNetRegressionModel()
    elif model_name == "MobileNetV3LRegressionModel":
        model = MobileNetV3LRegressionModel()

    train_model(model,
                train_loader,
                val_loader,
                epochs=10,
                learning_rate=0.001)
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    train(args.model_name,
          args.data_dir,
          args.batch_size,
          args.output_model_path)