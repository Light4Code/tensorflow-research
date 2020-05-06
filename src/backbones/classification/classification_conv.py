from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Model

from backbones import BaseBackbone
from utils import get_activation_layer


class ClassificationConv(BaseBackbone):
    def __init__(
        self,
        input_shape,
        hidden_activation,
        output_activation,
        leaky_alpha=0.1,
        num_classes: int = 2,
    ):
        super().__init__(
            input_shape,
            hidden_activation="relu",
            output_activation="softmax",
            leaky_alpha=leaky_alpha,
        )

        inputs = Input(shape=input_shape, name=self.input_name)

        x = Conv2D(32, (3, 3))(inputs)
        x = get_activation_layer(hidden_activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3))(x)
        x = get_activation_layer(hidden_activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3))(x)
        x = get_activation_layer(hidden_activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(32, (3, 3))(x)
        x = get_activation_layer(hidden_activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(16, (3, 3))(x)
        x = get_activation_layer(hidden_activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(64)(x)
        x = get_activation_layer(hidden_activation)(x)
        x = Dense(num_classes)(x)
        x = get_activation_layer(output_activation)(x)

        self.model = Model(inputs, x)
