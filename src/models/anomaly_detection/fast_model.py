import tensorflow as tf
import numpy as np
import numpy.ma as ma
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from models.base_model import BaseModel
import matplotlib.pyplot as plt


class FastModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def create_optimizer(self, optimzer="adam"):
        super().create_optimizer(optimzer)

    def create_model(self):
        input_shape = self.config.input_shape

        try:
            fast_model = self.config.train["fast_model"]
            translator_layer_size = fast_model["translator_layer_size"]
            middle_layer_size = fast_model["middle_layer_size"]
        except:
            translator_layer_size = 100
            middle_layer_size = 16

        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        input = tf.keras.Input(input_shape, name=self.input_name)
        x = input
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(translator_layer_size, activation="relu", name="encoder")(x)
        x = Dense(middle_layer_size, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(translator_layer_size, activation="relu", name="decoder")(x)
        x = Dense(input_dim, activation="sigmoid", name="reconstructor")(x)
        x = Reshape(input_shape)(x)

        self.model = Model(input, x)
        return self.model

    def compile(self, loss="mean_squared_error"):
        self.model.compile(loss=loss, optimizer=self.optimizer, metrics=["accuracy"])

    def plot_predictions(self, test_images):
        pred_count = len(self.predictions)
        plt_shape = (self.config.input_shape[0], self.config.input_shape[1])
        plt_cmap = "gray"
        if self.config.input_shape[2] > 1:
            plt_shape = (
                self.config.input_shape[0],
                self.config.input_shape[1],
                self.config.input_shape[2],
            )
        index = 1
        plt_index = 0
        for test_image in test_images:
            original_image = test_image.reshape(plt_shape)
            pred_image = self.predictions[plt_index].reshape(plt_shape)
            diff = self.image_util.create_diff(original_image, pred_image)
            mask = ma.masked_where(diff < self.config.test_threshold, diff)
            plt.subplot(pred_count, 4, index)
            plt.title("Original")
            plt.imshow(original_image, interpolation="none", cmap=plt_cmap)
            index += 1
            plt.subplot(pred_count, 4, index)
            plt.title("Prediction")
            plt.imshow(pred_image, interpolation="none", cmap=plt_cmap)
            index += 1
            plt.subplot(pred_count, 4, index)
            plt.title("Difference")
            plt.imshow(diff, interpolation="none", cmap=plt_cmap)
            index += 1
            plt.subplot(pred_count, 4, index)
            plt.title("Overlay")
            plt.imshow(original_image, interpolation="none", cmap=plt_cmap)
            plt.imshow(mask, cmap="jet", interpolation="none", alpha=0.7)
            index += 1
            plt_index += 1
        plt.show()
