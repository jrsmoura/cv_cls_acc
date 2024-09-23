import torch
import torch.nn as nn
from torchvision import models


class VGG16RegressionModel(nn.Module):
    """
    A regression model based on the VGG16 architecture.

    This model uses the VGG16 architecture pre-trained on ImageNet as a feature extractor,
    followed by custom layers for regression.

    Attributes:
        base_model (nn.Module): The VGG16 model without the top classification layers.
        global_avg (nn.AdaptiveAvgPool2d): Global average pooling layer.
        dense (nn.Linear): Dense layer with ReLU activation.
        relu (nn.ReLU): ReLU activation function.
        output_layer (nn.Linear): Dense layer for the output.
    """

    def __init__(self):
        """
        Initializes the VGG16RegressionModel.

        The VGG16 base model is initialized with pre-trained ImageNet weights and
        its top layers are removed. Custom layers for regression are added on top.
        """
        super(VGG16RegressionModel, self).__init__()
        self.base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing image data.

        Returns:
            torch.Tensor: Output tensor containing the regression result.
        """
        x = self.base_model(x)
        x = self.global_avg(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.dense(x))
        x = self.output_layer(x)
        return x


class ResNet50RegressionModel(nn.Module):
    """
    A regression model based on the ResNet50 architecture.

    This model uses the ResNet50 architecture pre-trained on ImageNet as a feature extractor,
    followed by custom layers for regression.

    Attributes:
        base_model (nn.Module): The ResNet50 model without the top classification layers.
        global_avg (nn.AdaptiveAvgPool2d): Global average pooling layer.
        dense (nn.Linear): Dense layer with ReLU activation.
        relu (nn.ReLU): ReLU activation function.
        output_layer (nn.Linear): Dense layer for the output.
    """

    def __init__(self):
        """
        Initializes the ResNet50RegressionModel.

        The ResNet50 base model is initialized with pre-trained ImageNet weights and
        its top layers are removed. Custom layers for regression are added on top.
        """
        super(ResNet50RegressionModel, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(2048, 256)  # ResNet50 output channels are 2048
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing image data.

        Returns:
            torch.Tensor: Output tensor containing the regression result.
        """
        x = self.base_model(x)
        x = self.global_avg(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.dense(x))
        x = self.output_layer(x)
        return x


class EfficientNetRegressionModel(nn.Module):
    """
    A regression model based on the EfficientNet architecture.

    This model uses the EfficientNet architecture pre-trained on ImageNet as a feature extractor,
    followed by custom layers for regression.

    Attributes:
        base_model (nn.Module): The EfficientNet model without the top classification layers.
        global_avg (nn.AdaptiveAvgPool2d): Global average pooling layer.
        dense (nn.Linear): Dense layer with ReLU activation.
        relu (nn.ReLU): ReLU activation function.
        output_layer (nn.Linear): Dense layer for the output.
    """

    def __init__(self):
        """
        Initializes the EfficientNetRegressionModel.

        The EfficientNet base model is initialized with pre-trained ImageNet weights and
        its top layers are removed. Custom layers for regression are added on top.
        """
        super(EfficientNetRegressionModel, self).__init__()
        self.base_model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(1280, 256)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing image data.

        Returns:
            torch.Tensor: Output tensor containing the regression result.
        """
        x = self.base_model(x)
        x = self.global_avg(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.dense(x))
        x = self.output_layer(x)
        return x


class InceptionV3RegressionModel(nn.Module):
    """
    A regression model based on the InceptionV3 architecture.

    This model uses the InceptionV3 architecture pre-trained on ImageNet as a feature extractor,
    followed by custom layers for regression.

    Attributes:
        base_model (nn.Module): The InceptionV3 model without the top classification layers.
        global_avg (nn.AdaptiveAvgPool2d): Global average pooling layer.
        dense (nn.Linear): Dense layer with ReLU activation.
        relu (nn.ReLU): ReLU activation function.
        output_layer (nn.Linear): Dense layer for the output.
    """

    def __init__(self):
        """
        Initializes the InceptionV3RegressionModel.

        The InceptionV3 base model is initialized with pre-trained ImageNet
        weights and
        its top layers are removed. Custom layers for regression are added on top.
        """
        super(InceptionV3RegressionModel, self).__init__()
        self.base_model = models.efficientnet_b0(
            weights=models.Inception_V3_Weights.DEFAULT
        )
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(1280, 256)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing image data.

        Returns:
            torch.Tensor: Output tensor containing the regression result.
        """
        x = self.base_model(x)
        x = self.global_avg(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.dense(x))
        x = self.output_layer(x)
        return x


class MobileNetV3LRegressionModel(nn.Module):
    """
    A regression model based on the  MobileNetV3 Large architecture.

    This model uses the  MobileNetV3 Large architecture pre-trained on ImageNet as a feature extractor,
    followed by custom layers for regression.

    Attributes:
        base_model (nn.Module): The  MobileNetV3 Large model without the top classification layers.
        global_avg (nn.AdaptiveAvgPool2d): Global average pooling layer.
        dense (nn.Linear): Dense layer with ReLU activation.
        relu (nn.ReLU): ReLU activation function.
        output_layer (nn.Linear): Dense layer for the output.
    """

    def __init__(self):
        """
        Initializes the MobileV3LRegressionModel.

        The MobileNetV3 Large base model is initialized with pre-trained
        ImageNet
        weights and
        its top layers are removed. Custom layers for regression are added on top.
        """
        super(MobileNetV3LRegressionModel, self).__init__()
        self.base_model = models.mobilenet_v3_large(weights=models.mobilenet_v3_large)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(1280, 256)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing image data.

        Returns:
            torch.Tensor: Output tensor containing the regression result.
        """
        x = self.base_model(x)
        x = self.global_avg(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.dense(x))
        x = self.output_layer(x)
        return x
