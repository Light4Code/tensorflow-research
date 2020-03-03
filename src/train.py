import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import *

from models.anomaly_detection.advanced_model import AdvancedModel
from models.anomaly_detection.fast_model import FastModel
from utils.config import Config
from utils.image_util import ImageUtil


def main():
    parser = argparse.ArgumentParser(
        description='Used to train TensorFlow model')
    parser.add_argument(
        'config',
        metavar="config",
        help='Path to the configuration file containing all parameters for model training'
    )
    parser.add_argument(
        "--train_files_path",
        dest='train_files_path',
        metavar="path",
        help='Overwrites the path included in the config to the training files'
    )
    parser.add_argument(
        "--test_file_path",
        dest='test_file_path',
        metavar="path",
        help='Overwrites the path included in the config to the test file'
    )
    parser.add_argument(
        "--test_threshold",
        dest='test_threshold',
        metavar="number (0-1)",
        type=float,
        help='Overwrites the test threshold'
    )
    parser.add_argument(
        "--epochs",
        dest='epochs',
        metavar="number (1-n)",
        type=int,
        help='Overwrites the train epochs'
    )
    parser.add_argument(
        "--optimizer",
        dest='optimizer',
        metavar="string (e.g. 'adam')",
        help='Overwrites the train optimizer'
    )
    parser.add_argument(
        "--loss",
        dest='loss',
        metavar="string (e.g. 'mse')",
        help='Overwrites the train loss parameter'
    )
    parser.add_argument(
        "--model",
        dest='model',
        metavar="string (e.g. 'advanced')",
        help='Overwrites the train model'
    )
    parser.add_argument(
        "--batch_size",
        dest='batch_size',
        metavar="number (1-n)",
        type=int,
        help='Overwrites the train batch size'
    )
    parser.add_argument(
        "--learning_rate",
        dest='learning_rate',
        metavar="number",
        type=float,
        help='Overwrites the learning rate'
    )

    args = parser.parse_args()
    config_path = args.config
    config = Config(config_path)
    image_util = ImageUtil()

    # Overwrite config
    if args.train_files_path:
        config.train_files_path = args.train_files_path
    if args.test_file_path:
        config.test_file_path = args.test_file_path
    if args.test_threshold:
        config.test_threshold = args.test_threshold
    if args.epochs:
        config.epochs = args.epochs
    if args.optimizer:
        config.optimizer = args.optimizer
    if args.loss:
        config.loss = args.loss
    if args.model:
        config.model = args.model
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate

    physical_devices = tf.config.list_physical_devices('GPU') 
    try: 
        tf.config.experimental.set_memory_growth(physical_devices[0], True) 
    except: 
        # Invalid device or cannot modify virtual devices once initialized. 
        pass 
    train(config, image_util)


def train(config, image_util):
    # Load training images
    train_images = load_images(config.train_files_path, config.input_shape, image_util)
    train_images = np.array(train_images)

    # Create train generator
    if config.image_data_generator:
        train_datagen = ImageDataGenerator(
            horizontal_flip=config.image_data_generator_horizonal_flip,
            fill_mode="nearest",
            zoom_range=config.image_data_generator_zoom_range,
            width_shift_range=config.image_data_generator_width_shift_range,
            height_shift_range=config.image_data_generator_height_shift_range,
            rotation_range=config.image_data_generator_rotation_range
        )
        train_datagen.fit(train_images)

    # ToDo: Create model
    model = create_model(config)

    # ToDo: Train model
    model.train(config, train_images, train_datagen)

    # ToDo: Display sample prediction
    if config.test_file_path and config.test_threshold:
        test_image = load_image(config.test_file_path, config, image_util)
        test_images = np.array([test_image])
        prediction = model.predict(test_images)

        plt_shape = (config.input_shape[0], config.input_shape[1])
        plt_cmap = 'gray'
        if config.input_shape[2] > 1:
            plt_shape = (
                config.input_shape[0], config.input_shape[1], config.input_shape[2])

        plt.subplot(221)
        plt.title('Input image')
        plt.imshow(test_image.reshape(plt_shape), cmap=plt_cmap)
        plt.subplot(222)
        plt.title('Prediction image')
        plt.imshow(prediction.reshape(plt_shape), cmap=plt_cmap)
        plt.subplot(223)
        plt.title('Difference image (before threshold)')
        diff = image_util.create_diff(test_image, prediction)
        plt.imshow(diff.reshape(plt_shape), cmap=plt_cmap)
        plt.subplot(224)
        plt.title('Result image (after threshold)')
        plt.imshow(image_util.apply_threshold(
            diff, config.test_threshold).reshape(plt_shape), cmap=plt_cmap)
        plt.show()


def load_images(files_path, input_shape, image_util):
    mode = image_util.get_color_mode(input_shape[2])

    images = image_util.load_images(files_path, mode)
    resized = []
    for img in images:
        res = image_util.resize_image(
            img, input_shape[1], input_shape[0])
        res = image_util.normalize(res, input_shape)
        resized.append(res)
    return resized


def load_image(path, config, image_util):
    mode = image_util.get_color_mode(config.input_shape[2])
    image = image_util.load_image(path, mode)
    resized = image_util.resize_image(
        image, config.input_shape[1], config.input_shape[0])
    resized = image_util.normalize(resized, config.input_shape)
    return resized

def create_optimizer(config):
    if config.optimizer == 'adam':
        optimizer = Adam(lr=config.learning_rate)
    elif config.optimizer == 'sgd':
        optimizer = SGD(lr=config.learning_rate, momentum=0.9)
    elif config.optimizer == 'rmsprop':
        optimizer = RMSprop(lr=config.learning_rate)
        config.loss = ''
    else:
        ValueError
    return optimizer

def create_model(config):
    if config.model == 'fast':
        model_container = FastModel(config.learning_rate)
    elif config.model == 'advanced':
        model_container = AdvancedModel(config.learning_rate)

    model_container.create(input_shape=config.input_shape)

    if config.optimizer and config.optimizer != model_container.optimizer_name:
        optimizer = create_optimizer(config)
        model_container.overwrite_optimizer(optimizer)
    
    if config.loss:
        model_container.compile(config.loss)
    else:
        model_container.compile()

    return model_container


if __name__ == '__main__':
    main()
