import abc
import os
import re
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adadelta, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import utils.image_util as iu
from utils import *
from utils.image_data_generator import *
from utils.plots import *


class BaseModel:
    def __init__(self, config: Type[Config]):
        super().__init__()
        self.input_name = "input"
        self.output_name = "output"
        self.initial_epoch = 0
        self.config = config

        self.prepare_training()
        self.create_callbacks()
        if self.config.train.optimizer:
            self.create_optimizer(self.config.train.optimizer)
        else:
            self.create_optimizer()

        self.create_model()

        if self.config.train.loss:
            self.compile(loss=self.config.train.loss)
        else:
            self.compile()

        self.load_weights()

    @abc.abstractmethod
    def create_model(self):
        return

    def compile(self, loss=None):
        self.model.compile(loss=loss, optimizer=self.optimizer, metrics=["accuracy"])

    def create_optimizer(self, optimizer=None):
        if not optimizer:
            raise ValueError

        learning_rate = self.config.train.learning_rate
        decay = self.config.train.decay
        momentum = self.config.train.momentum
        if optimizer == "adam":
            self.optimizer = Adam(lr=learning_rate, decay=decay)
        elif optimizer == "sgd":
            self.optimizer = SGD(lr=learning_rate, decay=decay, momentum=momentum)
        elif optimizer == "adadelta":
            self.optimizer = Adadelta(lr=learning_rate, decay=decay)
        else:
            raise ValueError

        self.optimizer_name = optimizer

    def create_callbacks(self):
        self.callbacks = []
        if self.config.train.early_stopping:
            if self.config.train.early_stopping.val_loss_epochs > 0:
                self.callbacks.append(
                    EarlyStopping(
                        monitor="val_loss",
                        min_delta=0,
                        patience=self.config.train.early_stopping.val_loss_epochs,
                        mode="auto",
                        restore_best_weights=True,
                    )
                )
        if self.config.train.checkpoints_path:
            self.callbacks.append(
                ModelCheckpoint(
                    self.config.train.checkpoints_path + "/model-{epoch:04d}.ckpts",
                    save_freq=self.config.train.checkpoint_save_period
                    * len(self.train_images),
                    save_weights_only=True,
                    save_best_only=self.config.train.checkpoint_save_best_only,
                )
            )

    def load_weights(self):
        if self.config.train.checkpoint_path:
            self.model.load_weights(self.config.train.checkpoint_path)
            try:
                checkpoint_name = os.path.basename(self.config.train.checkpoint_path)
                last_epoch = re.findall(r"\b\d+\b", checkpoint_name)[0]
                self.initial_epoch = int(last_epoch)
                self.config.train.epochs += self.initial_epoch
            except:
                print(
                    "Could not read checkpoint epoch information, will continue with epoch 0!"
                )

    def train(self):
        self.history = self.model.fit(
            self.train_images,
            self.y_train,
            batch_size=self.config.train.batch_size,
            epochs=self.config.train.epochs,
            callbacks=self.callbacks,
            shuffle=True,
            initial_epoch=self.initial_epoch,
            validation_split=self.config.train.validation_split,
        )

    def predict(self, test_images: []):
        self.predictions = []
        for img in test_images:
            self.predictions.append(
                self.model.predict(np.array([img], dtype=np.float32), batch_size=1)
            )

    def plot_predictions(self, test_images: []):
        plot_prediction(self.predictions, test_images, self.config.input_shape, self.config.eval.threshold)

    def prepare_training(self):
        self.train_images = None
        self.y_train = None
        self.train_images = self.load_images(
            self.config.train.files_path, self.config.input_shape
        )
        self.train_images = np.array(self.train_images, dtype=np.float32)
        if self.config.train.mask_files_path:
            original_masks = iu.create_mask_images(self.config)
            masks = []
            for m in original_masks:
                m = iu.normalize(m, self.config.input_shape)
                masks.append(m)
            self.y_train = masks
        else:
            self.y_train = self.train_images

        self.y_train = np.array(self.y_train, dtype=np.float32)

        # Create train generator
        self.train_images, self.y_train = self.generate_datagen(
            self.train_images, self.y_train
        )

    def generate_datagen(self, x: [], y: []):
        train_datagen = None
        y_train_datagen = None
        if self.config.train.image_data_generator:
            classes = []
            for _ in range(len(x)):
                classes.append(0)
            seed = 33

            train_datagen = create_image_data_generator(
                self.config.train.image_data_generator
            )
            y_train_datagen = create_image_data_generator(
                self.config.train.image_data_generator
            )

            train_datagen.fit(x, augment=True, seed=seed)
            y_train_datagen.fit(y, augment=True, seed=seed)

            train_gen = train_datagen.flow(x, batch_size=1, seed=seed,)
            train_y_gen = y_train_datagen.flow(y, batch_size=1, seed=seed,)

            tmp_train = []
            tmp_y = []
            for loop in range(self.config.train.image_data_generator.loop_count):
                for t in range(len(x)):
                    i = train_gen.next()
                    y = train_y_gen.next()
                    tmp_train.append(i[0])
                    tmp_y.append(y[0])

            return np.array(tmp_train), np.array(tmp_y)
        else:
            return np.array(x), np.array(y)

    def load_image(self, path: str, target_shape=(256, 256, 1)):
        mode = iu.get_color_mode(target_shape[2])
        image = iu.load_image(path, mode)
        resized = iu.resize_image(image, target_shape[1], target_shape[0])
        resized = iu.normalize(resized, target_shape)
        return resized

    def load_images(self, path: str, target_shape=(256, 256, 1)):
        mode = iu.get_color_mode(target_shape[2])
        images = iu.load_images(path, mode)
        resized = []
        for img in images:
            res = iu.resize_image(img, target_shape[1], target_shape[0])
            res = iu.normalize(res, target_shape)
            resized.append(res)
        return resized

    def plot_history(self):
        history = self.history
        plt.figure(figsize=(20, 10))
        plt.subplot(211)
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.subplot(212)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()
