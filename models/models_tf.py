import keras
import tensorflow as tf


class VGG16RegressionModel(keras.Model):
    """
    A regression model based on the VGG16 architecture.

    This model uses the VGG16 architecture pre-trained on ImageNet as a feature extractor,
    followed by custom layers for regression.

    Attributes:
        base_model (keras.Model): The VGG16 model without the top classification layers.
        global_avg (keras.layers.GlobalAveragePooling2D): Global average pooling layer.
        dense (keras.layers.Dense): Dense layer with ReLU activation.
        output_layer (keras.layers.Dense): Dense layer for the output.
    """

    def __init__(self) -> None:
        """
        Initializes the VGG16RegressionModel.

        The VGG16 base model is initialized with pre-trained ImageNet weights and
        its top layers are removed. Custom layers for regression are added on top.
        """
        super(VGG16RegressionModel, self).__init__()
        self.base_model = keras.applications.VGG16(
            weights='imagenet',
            include_top=False)
        self.base_model.trainable = False
        self.global_avg = keras.layers.GlobalAveragePooling2D()
        self.dense = keras.layers.Dense(256, activation='relu')
        self.output_layer = keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs (tf.Tensor): Input tensor containing image data.

        Returns:
            tf.Tensor: Output tensor containing the regression result.
        """
        x = self.base_model(inputs)
        x = self.global_avg(x)
        x = self.dense(x)
        return self.output_layer(x)

