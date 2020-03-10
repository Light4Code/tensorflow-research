import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Flatten,
                                     LeakyReLU, Reshape)
from tensorflow.keras.models import Model

from models import BaseModel
from utils.plots import *


class CustomConvAutoencoderModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def create_optimizer(self, optimzer="adam"):
        super().create_optimizer(optimzer)

    def compile(self, loss="mse"):
        self.model.compile(loss=loss, optimizer=self.optimizer, metrics=["accuracy"])

    def plot_predictions(self, test_images):
        plot_difference(self.config, self.predictions, test_images)
        
    def create_model(self):
        filters = (32, 64)
        kernel_size = (3,3)
        latent_dim = 16
        leak_alpha = 0.1
        try:
            model_config = self.config.train.raw["custom_conv_autoencoder_model"]
            latent_dim = model_config["latent_dim"]
            leak_alpha = model_config["leak_alpha"]
            filter_count = model_config["filters"]
            filter_size = model_config["filter_size"]
            filters = (filter_count, filter_size)
            kernel_size = (model_config["kernel_size"],model_config["kernel_size"])
        except:
            pass
            

        input_shape = self.config.input_shape
        inputs = Input(shape=input_shape, name=self.input_name)
        x = inputs
        for f in filters:
            x = Conv2D(filters=f, kernel_size=kernel_size, strides=2, padding="same")(x)
            x = LeakyReLU(alpha=leak_alpha)(x)
            x = BatchNormalization(axis=input_shape[2])(x)
        volume_size = tf.keras.backend.int_shape(x)
        x = Flatten()(x)
        latent = Dense(units=latent_dim)(x)  # Encoded
        encoder = Model(inputs, latent, name="encoder")

        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(np.prod(volume_size[1:]))(latent_inputs)
        x = Reshape((volume_size[1], volume_size[2], volume_size[3]))(x)
        for f in filters[::-1]:
            x = Conv2DTranspose(
                filters=f, kernel_size=kernel_size, strides=2, padding="same"
            )(x)
            x = LeakyReLU(alpha=leak_alpha)(x)
            x = BatchNormalization(axis=input_shape[2])(x)
        x = Conv2DTranspose(filters=input_shape[2], kernel_size=kernel_size, padding="same")(
            x
        )
        outputs = Activation("sigmoid", name=self.output_name)(x)  # Decoded
        decoder = Model(latent_inputs, outputs, name="decoder")

        self.model = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
        return self.model
