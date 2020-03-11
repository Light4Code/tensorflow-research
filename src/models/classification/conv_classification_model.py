import glob
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Model

from models import BaseModel


class ConvClassificationModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)
  
  def create_optimizer(self, optimizer="adam"):
    super().create_optimizer(optimizer=optimizer)

  def compile(self, loss=None): #categorical_crossentropy
    super().compile(tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

  def plot_predictions(self, test_images):
        # plot_difference(self.config, self.predictions, test_images)
        for p in self.predictions:
          print(p)

  def create_model(self):
    input_shape = self.config.input_shape
    inputs = Input(shape=input_shape, name=self.input_name)
    number_of_classes = 2

    x = Conv2D(32, (3, 3))(inputs)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(number_of_classes)(x)
    x = Activation('softmax')(x)

    self.model = Model(inputs, x)

  def train(self):
    self.history = self.model.fit(
                self.train_images,
                self.y_train,
                batch_size=self.config.train.batch_size,
                epochs=self.config.train.epochs,
                callbacks=self.callbacks,
                shuffle=True,
                initial_epoch=self.initial_epoch,
                validation_data=(self.validation_images, self.validation_labels)
            )

  def create_callbacks(self):
    super().create_callbacks()
    self.callbacks.append(EarlyStopping(monitor="val_loss", min_delta=0, patience=10, mode="auto", restore_best_weights=True))
    self.callbacks.append(EarlyStopping(monitor="val_accuracy", min_delta=0, patience=10, mode="auto", restore_best_weights=True))

  def prepare_training(self):
    class_dirs = []
    for di in glob.glob(self.config.train.files_path + "/*"):
      if os.path.isdir(di):
        print("Class: {0}".format(os.path.basename(di)))
        class_dirs.append(di)
    class_count = len(class_dirs)
    print("Detected classes: {0}".format(class_count))
    class_images = []
    class_index = 0
    for di in class_dirs:
      class_name = os.path.basename(di)
      images = self.load_images(di, self.config.input_shape)
      class_images.append(dict(index=class_index, name=class_name, images=images))
      class_index += 1
    
    all_images = []
    all_classes = []
    for d in class_images:
      for img in d["images"]:
        all_images.append(img)
        all_classes.append(d["index"])
    all_classes = np.array(all_classes, dtype=np.uint8)
    all_images = np.array(all_images, dtype=np.float32)
    train_data, test_data, train_labels, test_labels = train_test_split(all_images, all_classes, test_size=self.config.train.validation_split, random_state=33)
    self.train_images = train_data
    self.y_train = train_labels
    self.validation_images = test_data
    self.validation_labels = test_labels
