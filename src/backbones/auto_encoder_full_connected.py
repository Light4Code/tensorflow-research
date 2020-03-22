from .base_backbone import BaseBackbone
from utils.custom_types import Vector
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, BatchNormalization, Dense, Reshape


class AutoEncoderFullConnected(BaseBackbone):
    def __init__(
        self,
        input_shape: Vector,
        hidden_activation: str = "relu",
        output_activation: str = "sigmoid",
        translator_layer_size: int = 100,
        middle_layer_size: int = 16,
    ):
        super().__init__(input_shape, hidden_activation, output_activation)
        sub_layer_size = int(translator_layer_size / 2)

        inputs = Input(self.input_shape, name=self.input_name)
        x = Flatten()(inputs)
        x = BatchNormalization()(x)
        x = Dense(translator_layer_size, activation=hidden_activation, name="encoder")(
            x
        )
        x = Dense(sub_layer_size, activation=hidden_activation)(x)
        x = Dense(middle_layer_size, activation=hidden_activation)(x)
        x = Dense(sub_layer_size, activation=hidden_activation)(x)
        x = BatchNormalization()(x)
        x = Dense(translator_layer_size, activation=hidden_activation, name="decoder")(
            x
        )
        x = Dense(self.input_dim, activation=output_activation, name="reconstructor")(x)
        x = Reshape(input_shape, name=self.output_name)(x)
        self.model = Model(inputs, x)
