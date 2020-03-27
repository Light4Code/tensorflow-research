import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Flatten,
                                     LeakyReLU, Reshape)
from tensorflow.keras.models import Model

import backbones
from models import BaseModel
from utils.plots import *


class ConvAutoencoderModel(BaseModel):
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
        kernel_size = (3, 3)
        latent_dim = 16
        leak_alpha = 0.1
        try:
            model_config = self.config.train.raw["custom_conv_autoencoder_model"]
            latent_dim = model_config["latent_dim"]
            leak_alpha = model_config["leak_alpha"]
            filter_count = model_config["filters"]
            filter_size = model_config["filter_size"]
            filters = (filter_count, filter_size)
            kernel_size = (model_config["kernel_size"], model_config["kernel_size"])
        except:
            pass

        input_shape = self.config.input_shape
        backbone = backbones.AutoEncoderConv(input_shape)
        self.model = backbone.model
        return self.model
