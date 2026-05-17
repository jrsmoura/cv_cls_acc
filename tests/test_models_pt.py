import torch

from cv_cls_acc.models.models_pt import VGG16RegressionModel
from cv_cls_acc.models.models_pt import ResNet50RegressionModel
from cv_cls_acc.models.models_pt import MobileNetV3LRegressionModel


class TestVGG16:
    def test_instantiation(self):
        model = VGG16RegressionModel()
        assert model is not None

    def test_forward_output_shape(self):
        model = VGG16RegressionModel()
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 1)

    def test_backbone_is_frozen(self):
        model = VGG16RegressionModel()
        frozen = all(
            not p.requires_grad for p in model.base_model.parameters()
            )
        assert frozen

    def test_head_is_trainable(self):
        model = VGG16RegressionModel()
        head_params = (
            list(model.dense.parameters())
            + list(model.output_layer.parameters())
        )
        assert all(p.requires_grad for p in head_params)


class TestResNet50:
    def test_instantiation(self):
        model = ResNet50RegressionModel()
        assert model is not None

    def test_forward_output_shape(self):
        model = ResNet50RegressionModel()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 1)

    def test_backbone_is_frozen(self):
        model = ResNet50RegressionModel()
        frozen = all(
            not p.requires_grad for p in model.base_model.parameters()
            )
        assert frozen

    def test_head_is_trainable(self):
        model = ResNet50RegressionModel()
        head_params = (
            list(model.dense.parameters())
            + list(model.output_layer.parameters())
        )
        assert all(p.requires_grad for p in head_params)


class TestMobileNetV3:
    def test_instantiation(self):
        model = MobileNetV3LRegressionModel()
        assert model is not None

    def test_forward_output_shape(self):
        model = MobileNetV3LRegressionModel()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 1)

    def test_backbone_is_frozen(self):
        model = MobileNetV3LRegressionModel()
        frozen = all(
            not p.requires_grad for p in model.base_model.parameters()
        )
        assert frozen

    def test_head_is_trainable(self):
        model = MobileNetV3LRegressionModel()
        head_params = (
            list(model.dense.parameters())
            + list(model.output_layer.parameters())
        )
        assert all(p.requires_grad for p in head_params)
