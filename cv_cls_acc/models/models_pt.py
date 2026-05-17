import torch
import torch.nn as nn
from torchvision import models


class VGG16RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT
        ).features
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        x = self.global_avg(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.dense(x))
        return self.output_layer(x)


class ResNet50RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )
        self.base_model = nn.Sequential(*list(base.children())[:-2])
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(2048, 256)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        x = self.global_avg(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.dense(x))
        return self.output_layer(x)


class EfficientNetRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        self.base_model = nn.Sequential(*list(base.children())[:-2])
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(1280, 256)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        x = self.global_avg(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.dense(x))
        return self.output_layer(x)


class InceptionV3RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT
        )
        self.base_model = nn.Sequential(*list(base.children())[:-2])
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(2048, 256)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        x = self.global_avg(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.dense(x))
        return self.output_layer(x)


class MobileNetV3LRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )
        self.base_model = nn.Sequential(*list(base.children())[:-2])
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(1280, 256)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        x = self.global_avg(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.dense(x))
        return self.output_layer(x)
