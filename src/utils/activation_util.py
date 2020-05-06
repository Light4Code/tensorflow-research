from tensorflow.keras.layers import Activation, LeakyReLU


def get_activation_layer(
    activation_name: str = "relu", name: str = "", alpha: float = 0.1
):
    if activation_name != "leakyrelu":
        return Activation(activation_name, name=name)
    else:
        return LeakyReLU(alpha=alpha, name=name)
