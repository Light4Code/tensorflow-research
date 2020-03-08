import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from models import BaseModel


class AdvancedModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def create_optimizer(self, optimzer="adam"):
        super().create_optimizer(optimzer)

    def create_model(self, filters=(32, 64), latent_dim=16):
        input_shape = self.config.input_shape
        inputs = Input(shape=input_shape, name=self.input_name)
        x = inputs
        for f in filters:
            x = Conv2D(filters=f, kernel_size=(3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
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
                filters=f, kernel_size=(3, 3), strides=2, padding="same"
            )(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=input_shape[2])(x)
        x = Conv2DTranspose(filters=input_shape[2], kernel_size=(3, 3), padding="same")(
            x
        )
        outputs = Activation("sigmoid", name=self.output_name)(x)  # Decoded
        decoder = Model(latent_inputs, outputs, name="decoder")

        self.model = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
        return self.model

    def overwrite_optimizer(self, optimizer, optimizer_name):
        self.optimzer = optimizer
        self.optimizer_name = optimizer_name

    def compile(self, loss="mse"):
        self.model.compile(loss=loss, optimizer=self.optimizer, metrics=["accuracy"])
