import os
import re
import numpy as np
from utils.custom_types import Vector
from backbones import BaseBackbone
from utils.config import ImageGeneratorConfig
from utils.image_data_generator import create_image_data_generator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.callbacks import ModelCheckpoint


class TrainEngine:
    def __init__(
        self,
        input_shape: Vector,
        model: Model,
        optimizer: Optimizer,
        loss: str = "mean_squared_error",
        epochs: int = 100,
        checkpoints_save_path: str = None,
        checkpoint_save_period: int = 10,
        last_checkpoint_path: str = None,
    ):
        self.input_shape: Vector = input_shape
        self.optimizer: Optimizer = optimizer
        self.loss: str = loss
        self.epochs: int = epochs
        self.checkpoints_save_path: str = checkpoints_save_path
        self.checkpoint_save_period: int = checkpoint_save_period
        self._initial_epoch: int = 0
        self._history = None

        self.model: Model = model
        self.model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        self._load_weights(last_checkpoint_path)

    def train(
        self,
        train_x,
        train_y,
        batch_size: int = 10,
        validation_split: float = 0.0,
        image_generator_config: ImageGeneratorConfig = None,
        is_augment_y_enabled: bool = True,
    ):
        self._create_callbacks(train_x)
        batches_x, batches_y = self._split_into_batches(train_x, train_y, batch_size)
        initial_epoch = self._initial_epoch
        for epoch in range(self.epochs):
            epochs = self._initial_epoch + 1
            for batch_idx in range(len(batches_x)):
                epoch_train_x, epoch_train_y = self._augment_data(
                    batches_x[batch_idx],
                    batches_y[batch_idx],
                    image_generator_config,
                    is_augment_y_enabled,
                )
                self._train(epoch_train_x, epoch_train_y, epochs, validation_split)
                metric = self._eval(epoch_train_x, epoch_train_y)
                try:
                    train_loss = self._history.history["loss"]
                    # accuracy = self._history.history["accuracy"]
                    print(
                        "Epoch {0}/{1}\ttrain_loss: {2}\teval_loss: {3}\teval_acc: {4}".format(
                            epoch + 1 + initial_epoch,
                            self.epochs + initial_epoch,
                            round(train_loss[0], 5),
                            round(metric[0], 5),
                            metric[1],
                        )
                    )
                except:
                    pass

        return self._history

    def _train(
        self, epoch_data_x, epoch_data_y, epochs, validation_split,
    ):
        batch_size = len(epoch_data_x)

        self._history = self.model.fit(
            epoch_data_x,
            epoch_data_y,
            batch_size=batch_size,
            epochs=epochs,
            initial_epoch=self._initial_epoch,
            shuffle=True,
            verbose=0,
            validation_split=validation_split,
            callbacks=self.callbacks,
        )
        self._initial_epoch = self._initial_epoch + 1

    def _eval(self, epoch_data_x, epoch_data_y):
        batch_size = len(epoch_data_x)
        x = self.model.evaluate(
            epoch_data_x, epoch_data_y, batch_size=batch_size, verbose=0
        )
        return x

    def _split_into_batches(self, train_x, train_y, batch_size: int):
        batches = len(train_x) / batch_size
        batches_x = np.split(train_x, batches)
        batches_y = np.split(train_y, batches)
        return batches_x, batches_y

    def _augment_data(
        self,
        origin_train_x,
        origin_train_y,
        image_generator_config: ImageGeneratorConfig,
        is_augment_y_enabled: bool,
    ):
        train_x = None
        train_y = None
        if image_generator_config == None:
            return origin_train_x, origin_train_y

        seed = 33
        train_x_datagen = create_image_data_generator(image_generator_config)
        train_x_datagen.fit(train_x, augment=True, seed=seed)
        train_x_flow = train_x_datagen.flow(train_x, batch_size=1, seed=seed)

        train_y_flow = None
        if is_augment_y_enabled:
            train_y_datagen = create_image_data_generator(image_generator_config)
            train_y_datagen.fit(train_y, augment=True, seed=seed)
            train_y_flow = train_y_datagen.flow(train_y, batch_size=1, seed=seed)

        tmp_train_x = []
        tmp_train_y = []
        for loop in range(image_generator_config.loop_count):
            for x in range(len(train_x)):
                tmp_train_x.append(train_x_flow.next()[0])

                if not train_y_flow == None:
                    tmp_train_y.append(train_y_flow.next()[0])
                else:
                    tmp_train_y.append(train_y[x])

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
