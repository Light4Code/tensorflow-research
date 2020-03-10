from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    LeakyReLU,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
)
from tensorflow.keras.models import Model

from models import BaseModel
from utils.plots import *


class ConvAutoencoderModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def create_optimizer(self, optimizer="adam"):
        super().create_optimizer(optimizer=optimizer)

    def compile(self, loss="binary_crossentropy"):
        super().compile(loss=loss)

    def plot_predictions(self, test_images):
        plot_difference(self.config, self.predictions, test_images)

    def create_model(self):
        padding = "same"
        kernel_size = (5, 5)
        filters_size = 64
        filter_count = 2
        max_pooling_size = (2, 2)
        activation = "relu"
        activation_function = "sigmoid"
        input_shape = self.config.input_shape
        inputs = Input(shape=input_shape, name=self.input_name)
        x = inputs

        for _ in range(filter_count):
            x = Conv2D(
                filters_size, kernel_size, activation=activation, padding=padding
            )(x)
            x = MaxPooling2D(max_pooling_size, padding=padding)(x)

        for _ in range(filter_count):
            x = Conv2D(
                filters_size, kernel_size, activation=activation, padding=padding
            )(x)
            x = UpSampling2D(max_pooling_size)(x)

        decoded = Conv2D(
            1, kernel_size, activation=activation_function, padding=padding
        )(x)

        self.model = Model(inputs, decoded)

