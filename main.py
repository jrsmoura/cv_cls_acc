from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import config

from models.models_pt import (
    VGG16RegressionModel,
    ResNet50RegressionModel,
    EfficientNetRegressionModel,
    MobileNetV3LRegressionModel
)
from auxiliary_functions.aux_functions import get_filepaths_and_labels, \
    create_dataloader

if __name__ == '__main__':
    """
    Main training script for the regression models.

    This script initializes the dataset, creates dataloaders
    for value in iterable: passtraining and validation,
    and trains the chosen regression model (VGG16 or ResNet50).

    It uses Mean Squared Error (MSE) as the loss function and
    Adam as the optimizer. Training stops early if validation loss
    does not improve for a given patience period.
    """
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Retrieve file paths and labels for the dataset
    file_paths, labels = get_filepaths_and_labels(config.DATA_DIR)

    # Create dataloaders for training and validation
    train_loader, val_loader, train_size, val_size = create_dataloader(
        file_paths,
        labels,
        config.BATCH_SIZE
    )

    print(f"Train Size: {train_size}")
    print(f"Device: {config.DEVICE}")

    # Initialize the model, loss function, and optimizer
    model = VGG16RegressionModel().to(config.DEVICE)
    # model = ResNet50RegressionModel().to(config.DEVICE)
    # model = EfficientNetRegressionModel().to(config.DEVICE)
    # model = MobileNetV3LRegressionModel().to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    num_epochs = config.EPOCHS
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    print("Starting the training:")
    print("================================================")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.DEVICE), labels.to(
                    config.DEVICE)

                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)

                val_loss += loss.item() * images.size(0)

        val_loss /= val_size

        print(
            f'Epoch {epoch + 1}/{num_epochs},
            Train Loss: {train_loss:.4f},
            Val Loss: {val_loss:.4f}')

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
