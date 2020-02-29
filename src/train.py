from utils.config import Config
from utils.image_util import ImageUtil
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Used to train TensorFlow model')
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

    args = parser.parse_args()
    config_path = args.config
    config = Config(config_path)

    # Overwrite config
    if args.train_files_path:
        config.train_files_path = args.train_files_path

    # Load training images
    train_images = load_images(config)

    # ToDo: Create model

    # ToDo: Train model

    # ToDo: Eval model

    # ToDo: Display sample prediction

def load_images(config):
    iu = ImageUtil()
    mode = iu.cv2_grayscale
    if config.input_bpp == 3:
        mode = iu.cv2_color
    images = iu.load_images(config.train_files_path, mode)
    resized = []
    for img in images:
        resized.append(iu.resize_image(img, config.input_width, config.input_height))
    return resized


if __name__ == '__main__':
    main()
