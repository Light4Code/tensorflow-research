import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class FastModel():
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.optimizer_name = 'adam'
        self.optimizer = Adam(lr=learning_rate)

    def create(self, input_shape=(256,256,1), translator_layer_size=100, middle_layer_size=16):
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        input = tf.keras.Input(input_shape, name='input_shape')
        x = input
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(translator_layer_size, activation='relu', name='encoder')(x)
        x = Dense(middle_layer_size, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(translator_layer_size, activation='relu', name='decoder')(x)
        x = Dense(input_dim, activation='sigmoid', name='reconstructor')(x)
        x = Reshape(input_shape)(x)
        
        self.model = Model(input, x)
        print(self.model.summary())
        return self.model

    def overwrite_optimizer(self, optimizer, optimizer_name):
        self.optimzer = optimizer
        self.optimizer_name = optimizer_name

    def compile(self, loss='mean_squared_error'):
        self.model.compile(loss=loss, optimizer=self.optimizer, metrics=['accuracy'])

    def train(self, config, train_images, train_datagen):
      if config.image_data_generator:
          self.model.fit(train_datagen.flow(train_images, train_images, batch_size=config.batch_size),
                              epochs=config.epochs, steps_per_epoch=len(train_images) / config.batch_size)
      else:
          self.model.fit(train_images, train_images,
                    batch_size=config.batch_size, epochs=config.epochs)

    def predict(self, test_images):
      return self.model.predict(test_images, batch_size=len(test_images))
