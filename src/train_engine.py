import os
import re

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

import utils.image_util as iu
from backbones import BaseBackbone
from utils import print_epoch_statistics
from utils.config import ImageGeneratorConfig
from utils.custom_types import Vector
from utils.image_data_generator import create_image_data_generator


class TrainEngine:
    def __init__(
        self,
        input_shape: Vector,
        model: Model,
        optimizer: Optimizer,
        loss: str = "mean_squared_error",
        checkpoints_save_path: str = None,
        checkpoint_save_period: int = 10,
        last_checkpoint_path: str = None,
    ):
        self.input_shape: Vector = input_shape
        self.optimizer: Optimizer = optimizer
        self.loss: str = loss
        self.checkpoints_save_path: str = checkpoints_save_path
        self.checkpoint_save_period: int = checkpoint_save_period
        self._initial_epoch: int = 0
        self._train_history = None

        self.model: Model = model
        self.model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        self._load_weights(last_checkpoint_path)

        self.train_x_flow = None
        self.train_y_flow = None

    def train(
        self,
        train_x,
        train_y,
        eval_x,
        eval_y,
        batch_size: int = 32,
        epochs: int = 100,
        image_generator_config: ImageGeneratorConfig = None,
        is_augment_y_enabled: bool = True,
        augment_period: int = 10
    ):
        self._create_callbacks(train_x)
        initial_epoch = self._initial_epoch
        
        executed_epochs = 0
        hist_epochs = []
        hist_loss = []
        hist_acc = []
        hist_val_loss = []
        hist_val_acc = []
        while executed_epochs < epochs:
            # for idx in range(len(train_x)):
            #     m = iu.draw_mask(train_x[idx], train_y[idx], self.input_shape)
            #     iu.save_image(m, "D:/tmp/{0}.png".format(idx))

            epoch_train_x, epoch_train_y = self._augment_data(
                    train_x,
                    train_y,
                    image_generator_config,
                    is_augment_y_enabled,
                )

            for idx in range(len(epoch_train_x)):
                m = iu.draw_mask(epoch_train_x[idx], epoch_train_y[idx], self.input_shape)
                iu.save_image(m, "D:/tmp/{0}.png".format(idx))

            # Train
            step_epochs = self._initial_epoch + augment_period
            history = self._train(epoch_train_x, epoch_train_y, eval_x, eval_y, batch_size=batch_size, epochs=step_epochs)
            hist_epochs.extend(history.epoch)
            hist_loss.extend(history.history["loss"])
            hist_acc.extend(history.history["accuracy"])
            hist_val_loss.extend(history.history["val_loss"])
            hist_val_acc.extend(history.history["val_accuracy"])
            executed_epochs = executed_epochs + augment_period

            # Print epoch statistics
            print_epoch_statistics(hist_loss, hist_acc, hist_val_loss, hist_val_acc, executed_epochs, epochs, initial_epoch)

        return (hist_loss, hist_acc, hist_val_loss, hist_val_acc)

    def _train(
        self, train_x, train_y, eval_x, eval_y, batch_size: int, epochs: int,
    ):
        history= self.model.fit(
            train_x,
            train_y,
            batch_size=batch_size,
            epochs=epochs,
            initial_epoch=self._initial_epoch,
            shuffle=True,
            validation_data=(eval_x, eval_y),
            verbose=0,
        )
        self._initial_epoch = epochs
        return history

    def _eval(self, epoch_data_x, epoch_data_y):
        x = self.model.evaluate(
            epoch_data_x, epoch_data_y, batch_size=1, verbose=0
        )
        return x

    def _augment_data(
        self,
        origin_train_x,
        origin_train_y,
        image_generator_config: ImageGeneratorConfig,
        is_augment_y_enabled: bool,
    ):
        if image_generator_config == None:
            return origin_train_x, origin_train_y

        seed = 33
        if self.train_x_flow == None:
            train_x_datagen = create_image_data_generator(image_generator_config)
            train_x_datagen.fit(origin_train_x, augment=True, seed=seed)
            self.train_x_flow = train_x_datagen.flow(origin_train_x, batch_size=1, seed=seed)

        
        if is_augment_y_enabled and self.train_y_flow == None:
            train_y_datagen = create_image_data_generator(image_generator_config)
            train_y_datagen.fit(origin_train_y, augment=True, seed=seed)
            self.train_y_flow = train_y_datagen.flow(origin_train_y, batch_size=1, seed=seed)

        tmp_train_x = []
        tmp_train_y = []
        for _ in range(image_generator_config.loop_count):
            for x in range(len(origin_train_x)):
                tmp_train_x.append(self.train_x_flow.next()[0])

                if not self.train_y_flow == None:
                    tmp_train_y.append(self.train_y_flow.next()[0])
                else:
                    tmp_train_y.append(origin_train_y[x])

        return np.array(tmp_train_x), np.array(tmp_train_y)

    def _create_callbacks(self, train_x):
        self.callbacks = []
        if not self.checkpoints_save_path == None:
            self.callbacks.append(
                ModelCheckpoint(
                    self.checkpoints_save_path + "/model-{epoch:04d}.ckpts",
                    save_freq=self.checkpoint_save_period * len(train_x),
                    save_weights_only=True,
                )
            )

    def _load_weights(self, last_checkpoint_path: str):
        if not last_checkpoint_path == None:
            self.model.load_weights(last_checkpoint_path)
            try:
                checkpoint_name = os.path.basename(last_checkpoint_path)
                last_epoch = re.findall(r"\b\d+\b", checkpoint_name)[0]
                self._initial_epoch = int(last_epoch)
            except:
                print(
                    "Could not read checkpoint epoch information, will continue with epoch 0!"
                )
