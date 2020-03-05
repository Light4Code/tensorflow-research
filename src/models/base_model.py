from utils.config import Config
from tensorflow.keras.optimizers import Adam
from utils.image_util import ImageUtil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import abc

class BaseModel():
    def __init__(self, config):
        super().__init__()
        self.image_util = ImageUtil()
        self.config = config
        
        self.create_callbacks()
        self.create_model()
        self.create_optimizer()
        if self.config.loss:
            self.compile(loss=self.config.loss)
        else:
            self.compile()

    @abc.abstractmethod
    def create_model(self):
        return

    @abc.abstractmethod
    def compile(self):
        return

    @abc.abstractmethod
    def create_optimizer(self):
        return

    def create_callbacks(self):
        self.callbacks = []
        if self.config.checkpoints_path:
            self.callbacks.append(ModelCheckpoint(self.config.checkpoints_path + '/model-{epoch:04d}.ckpts', 
                                    save_freq=self.config.checkpoint_save_period * len(self.train_images), 
                                    save_weights_only=True,
                                    save_best_only=self.config.checkpoint_save_best_only))

    def train(self):
        if self.config.checkpoint_path:
            self.model.load_weights(self.config.checkpoint_path)

        if self.train_datagen:
            self.model.fit(self.train_datagen.flow(self.train_images, self.y_train, batch_size=self.config.batch_size, seed=33),
                                epochs=self.config.epochs, 
                                steps_per_epoch=len(self.train_images) / self.config.batch_size, 
                                callbacks=self.callbacks,
                                shuffle=True)
        else:
            self.model.fit(self.train_images, self.y_train,
                        batch_size=self.config.batch_size, 
                        epochs=self.config.epochs,
                        callbacks=self.callbacks,
                        shuffle=True)

    def predict(self):
        # return self.model.predict(test_images, batch_size=len(test_images))
        return

    def prepare_training(self):
        self.train_images = None
        self.y_train = None
        self.train_images = self.load_images(self.config.train_files_path, self.config.input_shape)
        self.train_images = np.array(self.train_images)
        if self.config.train_mask_files_path:
            self.y_train = self.load_images(self.config.train_mask_files_path, self.config.input_shape)
        else:
            self.y_train = self.train_images

        self.y_train = np.array(self.y_train)

        # Create train generator
        self.train_datagen = None
        if self.config.image_data_generator:
            self.train_datagen = ImageDataGenerator(
                horizontal_flip=self.config.image_data_generator_horizonal_flip,
                featurewise_center=self.config.image_data_generator_featurewise_center,
                featurewise_std_normalization=self.config.image_data_generator_featurewise_std_normalization,
                fill_mode="nearest",
                zoom_range=self.config.image_data_generator_zoom_range,
                width_shift_range=self.config.image_data_generator_width_shift_range,
                height_shift_range=self.config.image_data_generator_height_shift_range,
                rotation_range=self.config.image_data_generator_rotation_range
            )
            self.train_datagen.fit(train_images, augment=False, seed=33)
            

    def load_image(self, path, target_shape=(256,256,1)):
        mode = self.image_util.get_color_mode(target_shape[2])
        image = self.image_util.load_image(path, mode)
        resized = self.image_util.resize_image(
            image, target_shape[1], target_shape[0])
        resized = self.image_util.normalize(resized, target_shape)
        return resized
    
    def load_images(self, path, target_shape=(256,256,1)):
        mode = self.image_util.get_color_mode(target_shape[2])
        images = self.image_util.load_images(path, mode)
        resized = []
        for img in images:
            res = self.image_util.resize_image(
                img, target_shape[1], target_shape[0])
            res = self.image_util.normalize(res, target_shape)
            resized.append(res)
        return resized