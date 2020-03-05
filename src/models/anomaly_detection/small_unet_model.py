# implementation from: 
# https://github.com/Tony607/Industrial-Defect-Inspection-segmentation
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Input, Lambda,
                                     MaxPooling2D, UpSampling2D, concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def smooth_dice_coeff(smooth=1.):
    smooth = float(smooth)

    # IOU or dice coeff calculation
    def IOU_calc( y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def IOU_calc_loss(y_true, y_pred):
        return -IOU_calc(y_true, y_pred)

    return IOU_calc, IOU_calc_loss

IOU_calc, IOU_calc_loss = smooth_dice_coeff(0.00001)

class SmallUnetModel():
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.optimizer_name = 'adam'
        self.optimizer = Adam(lr=learning_rate)

    def create(self, config):
        input_shape = config.input_shape
        inputs = Input(shape=input_shape)
        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(64, kernel_size=(
            2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(32, kernel_size=(
            2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(16, kernel_size=(
            2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(8, kernel_size=(
            2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)

        return self.model

    def compile(self, loss=''):
      self.model.compile(optimizer=self.optimizer, loss=IOU_calc_loss, metrics=[IOU_calc])
    
    def train(self, config, train_images, train_datagen, train_mask_images):

        if config.checkpoint_path:
            self.model.load_weights(config.checkpoint_path)

        callbacks = []
        callbacks.append(ModelCheckpoint(config.checkpoints_path + '/model-{epoch:04d}.ckpts', 
                                save_freq=config.checkpoint_save_period * len(train_images), 
                                save_weights_only=True,
                                save_best_only=config.checkpoint_save_best_only))

        if config.image_data_generator:
            self.model.fit(train_datagen.flow(train_images, train_mask_images, batch_size=config.batch_size, seed=33),
                                epochs=config.epochs, 
                                steps_per_epoch=len(train_images) / config.batch_size, 
                                callbacks=callbacks,
                                shuffle=True)
        else:
            self.model.fit(train_images, train_mask_images,
                        batch_size=config.batch_size, 
                        epochs=config.epochs,
                        callbacks=callbacks,
                        shuffle=True)

    def predict(self, test_images):
      return self.model.predict(test_images, batch_size=len(test_images))
