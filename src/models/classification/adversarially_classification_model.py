import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    UpSampling2D,
    Flatten,
    Dense,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import neural_structured_learning as nsl
from models import BaseModel
from utils.plots import plot_classification


class AdversariallyClassificationModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def create_optimizer(self, optimizer=None):
        self.optimizer = Adam(lr=self.config.train.learning_rate)

    def compile(self, loss=None):
        self.adv_model.compile(
            loss="sparse_categorical_crossentropy", optimizer=self.optimizer,
        )
        self.model.compile(
            loss="sparse_categorical_crossentropy", optimizer=self.optimizer,
        )

    def create_model(self):
        input_shape = self.config.input_shape
        adv_config = nsl.configs.make_adv_reg_config(
            multiplier=0.2, adv_step_size=0.05, adv_grad_norm="infinity"
        )
        base_mode = self.build_base_model(input_shape=input_shape, num_classes=2)
        self.adv_model = nsl.keras.AdversarialRegularization(
            base_mode, label_keys=["label"], adv_config=adv_config
        )
        self.model = base_mode

    def train(self):
        ones = np.ones((len(self.train_images), 1), dtype=np.int)
        zeros = np.zeros((len(self.fake_images), 1), dtype=np.int)

        train_data = []
        train_data.extend(self.train_images)
        train_data.extend(self.fake_images)
        train_data = np.array(train_data)

        y_data = []
        y_data.extend(ones)
        y_data.extend(zeros)
        y_data = np.array(y_data, dtype=np.int)

        self.history = self.adv_model.fit(
            {"feature": train_data, "label": y_data},
            batch_size=self.config.train.batch_size,
            epochs=self.config.train.epochs,
            callbacks=self.callbacks,
            shuffle=True,
            initial_epoch=self.initial_epoch,
        )

    def build_base_model(
        self,
        input_shape=[28, 28, 1],
        num_classes=2,
        kernel_size=(3, 3),
        pool_size=(2, 2),
    ):
        inputs = Input(shape=input_shape, name=self.input_name)
        x = inputs

        x = Conv2D(32, kernel_size, activation="relu")(x)
        x = MaxPooling2D(pool_size=pool_size)(x)

        x = Conv2D(64, kernel_size, activation="relu")(x)
        x = MaxPooling2D(pool_size=pool_size)(x)

        x = Conv2D(64, kernel_size, activation="relu")(x)
        x = MaxPooling2D(pool_size=pool_size)(x)

        x = Conv2D(32, kernel_size, activation="relu")(x)
        x = MaxPooling2D(pool_size=pool_size)(x)

        x = Conv2D(16, kernel_size, activation="relu")(x)
        x = MaxPooling2D(pool_size=pool_size)(x)

        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs, x)
        return model

    def prepare_training(self):
        super().prepare_training()
        self.fake_images = self.load_images(
            self.config.train.alternative_files_path, self.config.input_shape
        )
        self.fake_images = np.array(self.fake_images, dtype=np.float32)
        x, y = self.generate_datagen(self.fake_images, self.fake_images)
        self.fake_images = x

    def plot_history(self):
        history = self.history
        plt.figure(figsize=(20, 10))
        plt.subplot(111)
        plt.plot(history.history["loss"])
        plt.plot(history.history["adversarial_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()

    def plot_predictions(self, test_images):
        plot_classification(self.config, self.predictions, test_images, ["NOK", "OK"])
