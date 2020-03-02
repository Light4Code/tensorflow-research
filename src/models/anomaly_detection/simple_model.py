import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class SimpleModel():
    def create(self, input_shape=(256,256,1)):
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        input = tf.keras.Input(input_shape, name='input_shape')
        x = input
        x = Flatten()(x)
        x = Dense(input_dim=input_dim, units = 100, activation='tanh')(x)
        x = Dense(units = 16, activation='tanh')(x)
        x = Dense(units = 100, activation='tanh')(x)
        x = Dense(units = input_dim, activation='tanh')(x)
        x = Reshape(input_shape)(x)
        
        model = Model(input, x)
        print(model.summary())
        return model
