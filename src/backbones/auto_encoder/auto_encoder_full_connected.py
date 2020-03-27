from tensorflow.keras import Input
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, Reshape
from tensorflow.keras.models import Model

from backbones.base_backbone import BaseBackbone
from utils.activation_util import get_activation_layer


class AutoEncoderFullConnected(BaseBackbone):
    def __init__(
        self,
        input_shape,
        hidden_activation="relu",
        output_activation="sigmoid",
        leaky_alpha=0.1,
        translator_layer_size: int = 100,
        middle_layer_size: int = 16,
    ):
        super().__init__(input_shape, hidden_activation, output_activation, leaky_alpha)
        sub_layer_size = int(translator_layer_size / 2)

        inputs = Input(self.input_shape, name=self.input_name)
        x = Flatten()(inputs)
        x = BatchNormalization()(x)
        x = Dense(translator_layer_size, name="encoder")(x)
        x = get_activation_layer(activation_name=hidden_activation, alpha=leaky_alpha)(
            x
        )
        x = Dense(sub_layer_size)(x)
        x = get_activation_layer(activation_name=hidden_activation, alpha=leaky_alpha)(
            x
        )
        x = Dense(middle_layer_size)(x)
        x = get_activation_layer(activation_name=hidden_activation, alpha=leaky_alpha)(
            x
        )
        x = Dense(sub_layer_size)(x)
        x = get_activation_layer(activation_name=hidden_activation, alpha=leaky_alpha)(
            x
        )
        x = BatchNormalization()(x)
        x = Dense(translator_layer_size, name="decoder")(x)
        x = get_activation_layer(activation_name=hidden_activation, alpha=leaky_alpha)(
            x
        )
        x = Dense(self.input_dim, name="reconstructor")(x)
        x = get_activation_layer(activation_name=output_activation, alpha=leaky_alpha)(
            x
        )
        x = Reshape(input_shape, name=self.output_name)(x)

        self.model = Model(inputs, x)
