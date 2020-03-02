import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class SimpleModel():
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.optimizer_name = 'adam'
        self.optimizer = Adam(lr=learning_rate)

    def create(self, input_shape=(256,256,1)):
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        input = tf.keras.Input(input_shape, name='input_shape')
        x = input
        x = Flatten()(x)
        x = Dense(input_dim=input_dim, units = 25, activation='relu')(x)
        x = Dense(units = 3, activation='relu')(x)
        x = Dense(units = 25, activation='relu')(x)
        x = Dense(units = input_dim)(x)
        x = Reshape(input_shape)(x)
        
        self.model = Model(input, x)
        print(self.model.summary())
        return self.model

    def overwrite_optimizer(self, optimizer, optimizer_name):
        self.optimzer = optimizer
        self.optimizer_name = optimizer_name

    def compile(self, loss='mean_squared_error'):
        self.model.compile(loss=loss, optimizer=self.optimizer, metrics=['accuracy'])
