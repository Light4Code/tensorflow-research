import abc
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import *
from utils.plots import *


class BaseModel:
    def __init__(self, config):
        super().__init__()
        self.input_name = "input"
        self.output_name = "output"
        self.initial_epoch = 0
        self.image_util = ImageUtil()
        self.config = config

        self.prepare_training()
        self.create_callbacks()
        self.create_model()
        if self.config.train.optimizer:
            self.create_optimizer(optimizer=self.config.train.optimizer)
        else:
            self.create_optimizer()

        if self.config.train.loss:
            self.compile(loss=self.config.train.loss)
        else:
            self.compile()
        self.load_weights()

    @abc.abstractmethod
    def create_model(self):
        return

    @abc.abstractmethod
    def compile(self, loss=None):
        return

    def create_optimizer(self, optimizer=None):
        if not optimizer:
            ValueError

        if optimizer == "adam":
            self.optimizer = Adam(lr=self.config.train.learning_rate)
        elif optimizer == "sgd":
            self.optimizer = SGD(lr=self.config.train.learning_rate, momentum=0.99)
        else:
            ValueError

        self.optimizer_name = optimizer

    def create_callbacks(self):
        self.callbacks = []
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
        if not self.train_datagen == None:
            self.history = self.model.fit(
                self.train_datagen,
                epochs=self.config.train.epochs,
                steps_per_epoch=len(self.train_images) / self.config.train.batch_size,
                callbacks=self.callbacks,
                shuffle=False,
                initial_epoch=self.initial_epoch,
                validation_data=self.valid_datagen
            )
            
        else:
            self.history = self.model.fit(
                self.train_images,
                self.y_train,
                batch_size=self.config.train.batch_size,
                epochs=self.config.train.epochs,
                callbacks=self.callbacks,
                shuffle=True,
                initial_epoch=self.initial_epoch,
                validation_split=self.config.train.validation_split
            )

    def predict(self, test_images):
        self.predictions = []
        for img in test_images:
            self.predictions.append(
                self.model.predict(np.array([img], dtype=np.float32), batch_size=1)
            )

    def plot_predictions(self, test_images):
        utils.plot_prediction(config, self.predictions, test_images)

    def prepare_training(self):
        self.train_images = None
        self.y_train = None
        self.train_images = self.load_images(
            self.config.train.files_path, self.config.input_shape
        )
        self.train_images = np.array(self.train_images, dtype=np.float32)
        if self.config.train.mask_files_path:
            original_masks = self.image_util.create_mask_images(self.config)
            masks = []
            for m in original_masks:
                m = self.image_util.normalize(m, self.config.input_shape)
                masks.append(m)
            self.y_train = masks
        else:
            self.y_train = self.train_images

        self.y_train = np.array(self.y_train, dtype=np.float32)

        # Create train generator
        self.generate_datagen()

    def generate_datagen(self):
        self.train_datagen = None
        self.y_train_datagen = None
        if self.config.train.image_data_generator:
            classes = []
            for _ in range(len(self.train_images)):
                classes.append(0)
            seed = 33
            self.train_datagen = ImageDataGenerator(
                     featurewise_center=self.config.image_data_generator_featurewise_center,
                     featurewise_std_normalization=self.config.image_data_generator_featurewise_std_normalization,
                     rotation_range=self.config.image_data_generator_rotation_range,
                     width_shift_range=self.config.image_data_generator_width_shift_range,
                     horizontal_flip=self.config.image_data_generator_horizonal_flip,
                     height_shift_range=self.config.image_data_generator_height_shift_range,
                     zoom_range=self.config.image_data_generator_zoom_range,
                     fill_mode='nearest',
                     validation_split=self.config.train.validation_split
            )
            self.y_train_datagen = ImageDataGenerator(
                     featurewise_center=self.config.image_data_generator_featurewise_center,
                     featurewise_std_normalization=self.config.image_data_generator_featurewise_std_normalization,
                     rotation_range=self.config.image_data_generator_rotation_range,
                     width_shift_range=self.config.image_data_generator_width_shift_range,
                     horizontal_flip=self.config.image_data_generator_horizonal_flip,
                     height_shift_range=self.config.image_data_generator_height_shift_range,
                     zoom_range=self.config.image_data_generator_zoom_range,
                     fill_mode='nearest',
                     validation_split=self.config.train.validation_split
            )
            self.train_datagen.fit(self.train_images, augment=True, seed=seed)
            self.y_train_datagen.fit(self.y_train, augment=True, seed=seed)
            self.train_datagen = self.train_datagen.flow(
                    self.train_images,
                    batch_size=self.config.train.batch_size,
                    seed=seed,
                    subset="training"
                    )
            y_train_gen = self.y_train_datagen.flow(
                    self.y_train,
                    batch_size=self.config.train.batch_size,
                    seed=seed,
                    subset="training"
                    )
            self.valid_datagen = self.y_train_datagen.flow(
                    self.train_images,
                    self.y_train,
                    batch_size=self.config.train.batch_size,
                    seed=seed,
                    subset="validation"
                    )

            self.train_datagen = self.create_image_iterator(self.train_datagen, y_train_gen)

    def create_image_iterator(self, train_data_generator, valid_data_generator):
        gen = zip(train_data_generator, valid_data_generator)
        for (img, val) in gen:
            yield (img, val)
            
    def load_image(self, path, target_shape=(256, 256, 1)):
        mode = self.image_util.get_color_mode(target_shape[2])
        image = self.image_util.load_image(path, mode)
        resized = self.image_util.resize_image(image, target_shape[1], target_shape[0])
        resized = self.image_util.normalize(resized, target_shape)
        return resized

    def load_images(self, path, target_shape=(256, 256, 1)):
        mode = self.image_util.get_color_mode(target_shape[2])
        images = self.image_util.load_images(path, mode)
        resized = []
        for img in images:
            res = self.image_util.resize_image(img, target_shape[1], target_shape[0])
            res = self.image_util.normalize(res, target_shape)
            resized.append(res)
        return resized
