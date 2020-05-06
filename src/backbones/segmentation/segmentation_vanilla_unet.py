from tensorflow.keras import Input
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dropout, Input,
                                     MaxPooling2D, concatenate)
from tensorflow.keras.models import Model

from backbones.base_backbone import BaseBackbone


class SegmentationVanillaUnet(BaseBackbone):
  def __init__(self, input_shape, hidden_activation="elu", output_activation="sigmoid", leaky_alpha=0.1, num_classes:int=1):
    super().__init__(input_shape, hidden_activation, output_activation, leaky_alpha=leaky_alpha)
    
    padding = "same"
    kernel_initializer = "he_normal"

    inputs = Input(shape=input_shape, name=self.input_name)

    conv1 = Conv2D(
            16,
            (3, 3),
            activation=hidden_activation,
            kernel_initializer=kernel_initializer,
            padding=padding,
        )(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(
        16,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(conv1)
    pooling1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(
        32,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(pooling1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(
        32,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(conv2)
    pooling2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(
        64,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(pooling2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(
        64,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(conv3)
    pooing3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(
        128,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(pooing3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(
        128,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(conv4)
    pooling4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(
        256,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(pooling4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(
        256,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(conv5)

    upconv6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding=padding)(conv5)
    upconv6 = concatenate([upconv6, conv4])
    conv6 = Conv2D(
        128,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(upconv6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(
        128,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(conv6)

    upconv7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding=padding)(conv6)
    upconv7 = concatenate([upconv7, conv3])
    conv7 = Conv2D(
        64,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(upconv7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(
        64,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(conv7)

    upconv8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding=padding)(conv7)
    upconv8 = concatenate([upconv8, conv2])
    conv8 = Conv2D(
        32,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(upconv8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(
        32,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(conv8)

    upconv9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding=padding)(conv8)
    upconv9 = concatenate([upconv9, conv1], axis=3)
    conv9 = Conv2D(
        16,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(upconv9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(
        16,
        (3, 3),
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(conv9)

    outputs = Conv2D(
        num_classes, (1, 1), activation=output_activation, name=self.output_name
    )(conv9)

    self.model = Model(inputs=[inputs], outputs=[outputs])
