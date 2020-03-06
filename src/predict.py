import argparse

import numpy as np

from utils.config import Config
from utils.image_util import ImageUtil
from utils.model_creater import create_model


def main():
    parser = argparse.ArgumentParser(
        description="Used to predict TensorFlow model checkpoint"
    )
    parser.add_argument(
        "config",
        metavar="config",
        help="Path to the configuration file containing all parameters for model training",
    )
    parser.add_argument(
        "--test_files_path",
        dest="test_files_path",
        metavar="path",
        help="Path to the test files that should be predicted",
    )
    parser.add_argument(
        "--model",
        dest="model",
        metavar="string (e.g. 'small', 'advanced', 'small_unet')",
        help="Overwrites the train model",
    )
    parser.add_argument(
        "--checkpoint_path",
        dest="checkpoint_path",
        metavar="path",
        help="Overwrites the path to the saved checkpoint containing the model weights",
    )

    args = parser.parse_args()
    config_path = args.config
    config = Config(config_path)
    image_util = ImageUtil()

    if args.model:
        config.model = args.model
    if args.checkpoint_path:
        config.checkpoint_path = args.checkpoint_path
    if args.test_files_path:
        config.test_files_path = args.test_files_path

    color_mode = image_util.cv2_grayscale
    if config.input_shape[2] == 3:
        color_mode = image_util.cv2_color

    test_images = image_util.load_images(config.test_files_path, color_mode)
    tmp_imgs = []
    for img in test_images:
        res = image_util.resize_image(img, config.input_shape[1], config.input_shape[0])
        norm = image_util.normalize(res, config.input_shape)
        tmp_imgs.append(norm)
    test_images = np.array(tmp_imgs)

    model = create_model(config)
    model.predict(test_images)
    model.plot_predictions(test_images)


if __name__ == "__main__":
    main()
