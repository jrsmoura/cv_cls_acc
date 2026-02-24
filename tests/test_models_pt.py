"""Tests for PyTorch regression models"""

import torch

from models.models_pt import VGG16RegressionModel


class TestVGG16:
    """Tests for VGG16RegressionModel"""

    def test_instantiation(self):
        """Model should instantiate without errors"""
        model = VGG16RegressionModel()
        assert model is not None

    def test_forward_output_shape(self):
        """Forward pass should return shape (batch_size, 1) for regression"""
        model = VGG16RegressionModel()
        model.eval()

        x = torch.randn(2, 3, 224, 224)  # (#imagens, cores, x_px, y_px)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 1)

    def test_backbone_is_frozen(self):
        """All backbone parameters must have requires_grad=false"""
        model = VGG16RegressionModel()
        frozen = all(
            not p.requires_grad for p in model.base_model.parameters())
        assert frozen, "Some backbone parameters are not frozen"

    def test_head_is_trainable(self):
        """Regression head parameters must be trainable."""
        model = VGG16RegressionModel()
        head_params = list(model.dense.parameters()) + list(
            model.output_layer.parameters()
        )
        trainable = all(p.requires_grad for p in head_params)
        assert trainable, "Some head parameters are not trainable."
