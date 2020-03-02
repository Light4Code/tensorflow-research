import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Flatten,
                                     LeakyReLU, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


class AdvancedModel():
  def __init__(self, learning_rate=1e-4):
    super().__init__()
    self.optimizer_name = 'sgd'
    self.optimizer = SGD(lr=learning_rate, momentum=0.9)

  def create(self, input_shape=(256,256,1), filters=(32, 64), latent_dim=16):
      inputs = Input(shape=input_shape)
      x = inputs
      for f in filters:
        x = Conv2D(filters=f, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=input_shape[2])(x)
      volume_size = tf.keras.backend.int_shape(x)
      x = Flatten()(x)
      latent = Dense(units=latent_dim)(x) # Encoded
      encoder = Model(inputs, latent, name='encoder')
      print(encoder.summary())

      latent_inputs = Input(shape=(latent_dim,))
      x = Dense(np.prod(volume_size[1:]))(latent_inputs)
      x = Reshape((volume_size[1], volume_size[2], volume_size[3]))(x)
      for f in filters[::-1]:
        x = Conv2DTranspose(filters=f, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=input_shape[2])(x)
      x = Conv2DTranspose(filters=input_shape[2], kernel_size=(3, 3), padding='same')(x)
      outputs = Activation('sigmoid')(x) # Decoded
      decoder = Model(latent_inputs, outputs, name='decoder')
      print(decoder.summary())

      self.autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
      print(self.autoencoder.summary())
      return self.autoencoder

  def overwrite_optimizer(self, optimizer, optimizer_name):
      self.optimzer = optimizer
      self.optimizer_name = optimizer_name

  def compile(self, loss='mse'):
      self.autoencoder.compile(loss=loss, optimizer=self.optimizer, metrics=['accuracy'])
    
