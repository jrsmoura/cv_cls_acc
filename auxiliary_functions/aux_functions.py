import os
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image

import config


def get_filepaths_and_labels(data_dir: str) -> Tuple[List[str], List[float]]:
    """
    Retrieves file paths and corresponding labels from a directory structure.

    Args:
        data_dir (str): Directory containing subdirectories of images, named by label.

    Returns:
        Tuple[List[str], List[float]]: Lists of image file paths and their corresponding labels.
    """
    file_paths: List[str] = []
    labels: List[float] = []
    ages = os.listdir(data_dir)
    for age in ages:
        age_dir = os.path.join(data_dir, str(age))
        if os.path.isdir(age_dir):
            for img_name in os.listdir(age_dir):
                file_paths.append(os.path.join(age_dir, img_name))
                age_float = float(age)
                labels.append(age_float)
    return file_paths, labels


def process_data_pt(image_path, label):
    """
    Processes an image and its label for PyTorch.

    Args:
        image_path (str): Path to the image file.
        label (float): Label associated with the image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Processed image tensor and its label.
    """
    image = read_image(image_path).float() / 255.0
    image = transforms.Resize(config.IMG_SIZE)(image)
    return image, torch.tensor(label, dtype=torch.float32)


def create_dataloader(file_paths, labels, batch_size, val_split=0.2):
    """
    Creates PyTorch DataLoaders for training and validation datasets.

    Args:
        file_paths (List[str]): List of image file paths.
        labels (List[float]): List of labels corresponding to the images.
        batch_size (int): Number of samples per batch.
        val_split (float): Fraction of data to reserve for validation.

    Returns:
        Tuple[DataLoader, DataLoader, int, int]: Training DataLoader, validation DataLoader, size of training set, size of validation set.
    """
    transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = CustomImageDataset(file_paths, labels, transform=transform)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2)

    return train_loader, val_loader, train_size, val_size


def save_model(epochs, model, optimizer, criterion, pretrained):
    """
    Saves the trained PyTorch model to disk.

    Args:
        epochs (int): Number of epochs the model was trained for.
        model (nn.Module): The trained model to be saved.
        optimizer (torch.optim.Optimizer): Optimizer used during training.
        criterion: Loss function used during training.
        pretrained (bool): Whether the model was pretrained or not.
    """
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, f"../outputs/model_pretrained_{pretrained}.pth")


class CustomImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        """
        Custom Dataset class for loading images and labels.

        Args:
            file_paths (List[str]): List of image file paths.
            labels (List[float]): List of labels corresponding to the images.
            transform (torchvision.transforms.Compose, optional): Transformations to be applied on the images.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Fetches the image and label corresponding to the given index.

        Args:
            idx (int): Index of the sample to be fetched.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image and its corresponding label.
        """
        image = read_image(self.file_paths[idx]).float() / 255.0
        if image.shape[0] != 3:
            image = image.expand(3, -1, -1)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label
