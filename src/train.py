from utils.config import Config
from utils.image_util import ImageUtil
from models.anomaly_detection.simple_model import SimpleModel
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt


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

    # Load training images
    train_images = load_images(config, image_util)
    train_images = np.array(train_images)

    # ToDo: Create model
    model = create_model(config)

    # ToDo: Train model
    model.fit(train_images, train_images,
              batch_size=config.batch_size, epochs=config.epochs)

    # ToDo: Eval model

    # ToDo: Display sample prediction
    if config.test_file_path and config.test_threshold:
        test_image = load_image(config.test_file_path, config, image_util)
        test_images = np.array([test_image])
        prediction = model.predict(test_images, batch_size=1)
        plt.subplot(221)
        plt.title('Input image')
        plt.imshow(test_image.reshape(config.input_shape[0], config.input_shape[1]), cmap='gray')
        plt.subplot(222)
        plt.title('Prediction image')
        plt.imshow(prediction.reshape(config.input_shape[0], config.input_shape[1]), cmap='gray')
        plt.subplot(223)
        plt.title('Difference image (before threshold)')
        diff = image_util.create_diff(test_image, prediction)
        plt.imshow(diff.reshape(config.input_shape[0], config.input_shape[1]), cmap='gray')
        plt.subplot(224)
        plt.title('Result image (after threshold)')
        plt.imshow(image_util.apply_threshold(diff, config.test_threshold).reshape(config.input_shape[0], config.input_shape[1]), cmap='gray')
        plt.show()


def load_images(config, image_util):
    mode = image_util.get_color_mode(config.input_shape[2])

    images = image_util.load_images(config.train_files_path, mode)
    resized = []
    for img in images:
        res = image_util.resize_image(
            img, config.input_shape[0], config.input_shape[1])
        res = image_util.normalize(res, config.input_shape)
        resized.append(res)
    return resized


def load_image(path, config, image_util):
    mode = image_util.get_color_mode(config.input_shape[2])
    image = image_util.load_image(path, mode)
    resized = image_util.resize_image(
        image, config.input_shape[0], config.input_shape[1])
    resized = image_util.normalize(resized, config.input_shape)
    return resized


def create_model(config):
    model = None
    model = SimpleModel()
    model = model.create(learning_rate=config.learning_rate,
                         input_shape=config.input_shape)

    return model


if __name__ == '__main__':
    main()
