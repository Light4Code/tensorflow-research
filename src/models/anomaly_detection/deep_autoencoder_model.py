import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models import BaseModel
from utils.plots import *
from backbones import AutoEncoderFullConnected


class DeepAutoencoderModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def create_optimizer(self, optimzer="adam"):
        super().create_optimizer(optimzer)

    def create_model(self):
        input_shape = self.config.input_shape

        try:
            model_config = self.config.train.raw["deep_autoencoder_model"]
            translator_layer_size = model_config["translator_layer_size"]
            middle_layer_size = model_config["middle_layer_size"]
        except:
            translator_layer_size = 100
            middle_layer_size = 16

        backbone = AutoEncoderFullConnected(
            input_shape,
            translator_layer_size=translator_layer_size,
            middle_layer_size=middle_layer_size,
        )
        self.model = backbone.model
        return backbone.model

    def compile(self, loss="mean_squared_error"):
        self.model.compile(loss=loss, optimizer=self.optimizer, metrics=["accuracy"])

    def plot_predictions(self, test_images):
        plot_difference(self.config, self.predictions, test_images)
