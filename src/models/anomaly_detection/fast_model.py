import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from models.base_model import BaseModel

class FastModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        

    def create_optimizer(self):
        self.optimizer_name = 'adam'
        self.optimizer = Adam(lr=self.config.learning_rate)

    def create_model(self):
        input_shape = self.config.input_shape

        try:
            fast_model = self.config.train['fast_model']
            translator_layer_size = fast_model['translator_layer_size']
            middle_layer_size = fast_model['middle_layer_size']
        except:
            translator_layer_size = 100
            middle_layer_size = 16

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
        return self.model

    def compile(self, loss='mean_squared_error'):
        self.model.compile(loss=loss, optimizer=self.optimizer, metrics=['accuracy'])
