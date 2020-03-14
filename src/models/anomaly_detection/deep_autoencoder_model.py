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

        sub_layer_size = int(translator_layer_size / 2)
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        inputs = tf.keras.Input(input_shape, name=self.input_name)
        x = inputs
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(translator_layer_size, activation="relu", name="encoder")(x)
        x = Dense(sub_layer_size, activation="relu")(x)
        x = Dense(middle_layer_size, activation="relu")(x)
        x = Dense(sub_layer_size, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(translator_layer_size, activation="relu", name="decoder")(x)
        x = Dense(input_dim, activation="sigmoid", name="reconstructor")(x)
        x = Reshape(input_shape, name=self.output_name)(x)
        self.model = Model(inputs, x)
        return self.model

    def compile(self, loss="mean_squared_error"):
        self.model.compile(loss=loss, optimizer=self.optimizer, metrics=["accuracy"])

    def plot_predictions(self, test_images):
        plot_difference(self.config, self.predictions, test_images)
