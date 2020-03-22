import argparse

import numpy as np

from utils import *
from tensorflow.keras import backend as K


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
        "--eval_files_path",
        dest="eval_files_path",
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

    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    args = parser.parse_args()
    config_path = args.config
    config = Config(config_path)
    image_util = ImageUtil()

    if args.model:
        config.model = args.model
    if args.checkpoint_path:
        config.train.checkpoint_path = args.checkpoint_path
    if args.eval_files_path:
        config.eval.files_path = args.eval_files_path

    color_mode = image_util.cv2_grayscale
    if config.input_shape[2] == 3:
        color_mode = image_util.cv2_color

    test_images = image_util.load_images(config.eval.files_path, color_mode)
    tmp_imgs = []
    for img in test_images:
        res = image_util.resize_image(img, config.input_shape[1], config.input_shape[0])
        norm = image_util.normalize(res, config.input_shape)
        tmp_imgs.append(norm)
    test_images = np.array(tmp_imgs, dtype=np.float32)

    model = create_model(config)
    model.predict(test_images)
    model.plot_predictions(test_images)

    K.clear_session()


if __name__ == "__main__":
    main()
