# implementation from:
# https://github.com/Tony607/Industrial-Defect-Inspection-segmentation
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Input,
    Lambda,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
)
from tensorflow.keras.models import Model
from models import BaseModel


def smooth_dice_coeff(smooth=1.0):
    smooth = float(smooth)

    # IOU or dice coeff calculation
    def IOU_calc(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (
            2 * (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        )

    def IOU_calc_loss(y_true, y_pred):
        return -IOU_calc(y_true, y_pred)

    return IOU_calc, IOU_calc_loss


IOU_calc, IOU_calc_loss = smooth_dice_coeff(0.00001)


class SmallUnetModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def create_optimizer(self, optimzer="adam"):
        super().create_optimizer(optimzer)

    def compile(self, loss=None):
        if loss:
            self.model.compile(
                optimizer=self.optimizer, loss=loss, metrics=[IOU_calc],
            )
        else:
            self.model.compile(
                optimizer=self.optimizer, loss=IOU_calc_loss, metrics=[IOU_calc]
            )

    def create_model(self):
        input_shape = self.config.input_shape
        inputs = Input(shape=input_shape, name=self.input_name)
        conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
        conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(pool1)
        conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool2)
        conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool3)
        conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool4)
        conv5 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv5)

        up6 = concatenate(
            [
                Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding="same")(
                    conv5
                ),
                conv4,
            ],
            axis=3,
        )
        conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(up6)
        conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv6)

        up7 = concatenate(
            [
                Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding="same")(
                    conv6
                ),
                conv3,
            ],
            axis=3,
        )
        conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")(up7)
        conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv7)

        up8 = concatenate(
            [
                Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), padding="same")(
                    conv7
                ),
                conv2,
            ],
            axis=3,
        )
        conv8 = Conv2D(32, (3, 3), activation="relu", padding="same")(up8)
        conv8 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv8)

        up9 = concatenate(
            [
                Conv2DTranspose(8, kernel_size=(2, 2), strides=(2, 2), padding="same")(
                    conv8
                ),
                conv1,
            ],
            axis=3,
        )
        conv9 = Conv2D(16, (3, 3), activation="relu", padding="same")(up9)
        conv9 = Conv2D(16, (3, 3), activation="relu", padding="same")(conv9)

        conv10 = Conv2D(1, (1, 1), activation="sigmoid", name=self.output_name)(conv9)

        self.model = Model(inputs=[inputs], outputs=[conv10])

        return self.model

