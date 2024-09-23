import numpy as np
from typing import Dict, List
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.models_pt import VGG16RegressionModel


def train_model(model, train_loader, criterion, optimizer, num_epochs: int = 5) -> None:
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {running_loss / len(train_loader):.4f}"
        )
    print("Finished Training")


def validate_model(model, val_loader, criterion) -> float:
    model.eval()
    val_loss: float = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def create_optimizer(model, opt: str, lr: float, momentum: float):
    if opt == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif opt == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif opt == "RMSProp":
        return optim.RMSprop(model.parameters(), lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {opt}")


def finetune_model(
    model,
    train_loader,
    criterion,
    num_epochs: int,
    learning_rates: np.ndarray,
    momentums: List[float],
    optimizers: List[str],
) -> Dict:
    best_loss: float = float("inf")
    best_params: dict = {}
    val_loss: float = 0.0
    for lr in learning_rates:
        for momentum in momentums:
            for opt in optimizers:
                optimizer = create_optimizer(model, opt, lr, momentum)
                print(f"Training with lr={lr}," f"momentum={momentum}, optimizer={opt}")
                train_model(model, train_loader, criterion, optimizer, num_epochs)

                val_loss = validate_model(model, val_loader, criterion)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = {"lr": lr, "momentum": momentum, "optimizer": opt}
                    torch.save(model.state_dict(), "best_model.pth")

    return best_params


transform = transforms.Compose(
    [
        transforms.Resize((240, 240)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 2.225]),
    ]
)

train_dataset = datasets.ImageFolder(root="../data/age-regression", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataset = ...
val_loader = ...
model = VGG16RegressionModel()
criterion = torch.nn.MSELoss()
num_epochs = 5

learning_rates = np.logspace(-5, -1, num=5)
momentums = [0.9, 0.95, 0.99]
optimizers = ["SGD", "Adam", "RMSProp", "AdamW"]

best_params = finetune_model(
    model, train_loader, criterion, num_epochs, learning_rates, momentums, optimizers
)
print("Best parameters:", best_params)
