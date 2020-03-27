import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Flatten, Reshape)
from tensorflow.keras.models import Model

from backbones.base_backbone import BaseBackbone
from utils.activation_util import get_activation_layer
from utils.custom_types import Vector


class AutoEncoderConv(BaseBackbone):
    def __init__(
        self,
        input_shape,
        hidden_activation="leakyrelu",
        output_activation="sigmoid",
        leaky_alpha=0.1,
        filters: Vector = (32, 64),
        kernel_size: Vector = (3, 3),
        strides: int = 2,
        padding: str = "same",
        latent_dim: int = 16,
    ):
        super().__init__(
            input_shape, hidden_activation, output_activation, leaky_alpha=leaky_alpha
        )
        channels = input_shape[2]

        inputs = Input(shape=input_shape, name=self.input_name)
        x = inputs

        for f in filters:
            x = Conv2D(
                filters=f, kernel_size=kernel_size, strides=strides, padding=padding
            )(x)
            x = get_activation_layer(activation_name=hidden_activation)(x)
            x = BatchNormalization(axis=channels)(x)
        volume_size = tf.keras.backend.int_shape(x)
        x = Flatten()(x)
        latent = Dense(units=latent_dim)(x)
        encoder = Model(inputs, latent, name="encoder")

        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(np.prod(volume_size[1:]))(latent_inputs)
        x = Reshape((volume_size[1], volume_size[2], volume_size[3]))(x)
        for f in filters[::-1]:
            x = Conv2DTranspose(
                filters=f, kernel_size=kernel_size, strides=strides, padding=padding
            )(x)
            x = get_activation_layer(activation_name=hidden_activation)(x)
            x = BatchNormalization(axis=channels)(x)
        x = Conv2DTranspose(filters=channels, kernel_size=kernel_size, padding=padding)(
            x
        )
        outputs = get_activation_layer(
            activation_name=output_activation, name=self.output_name
        )(x)
        decoder = Model(latent_inputs, outputs, name="decoder")

        self.model = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
