import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from utils.config import Config
from utils.model_creater import create_model


def main():
    parser = argparse.ArgumentParser(description="Used to train TensorFlow model")
    parser.add_argument(
        "config",
        metavar="config",
        help="Path to the configuration file containing all parameters for model training",
    )
    parser.add_argument(
        "--train_files_path",
        dest="train_files_path",
        metavar="path",
        help="Overwrites the path included in the config to the training files",
    )
    parser.add_argument(
        "--train_mask_files_path",
        dest="train_mask_files_path",
        metavar="path",
        help="Overwrites the path included in the config to the training mask files",
    )
    parser.add_argument(
        "--test_file_path",
        dest="test_file_path",
        metavar="path",
        help="Overwrites the path included in the config to the test file",
    )
    parser.add_argument(
        "--test_threshold",
        dest="test_threshold",
        metavar="number (0-1)",
        type=float,
        help="Overwrites the test threshold",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        metavar="number (1-n)",
        type=int,
        help="Overwrites the train epochs",
    )
    parser.add_argument(
        "--loss",
        dest="loss",
        metavar="string (e.g. 'mse')",
        help="Overwrites the train loss parameter",
    )
    parser.add_argument(
        "--model",
        dest="model",
        metavar="string (e.g. 'small', 'advanced', 'small_unet')",
        help="Overwrites the train model",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        metavar="number (1-n)",
        type=int,
        help="Overwrites the train batch size",
    )
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        metavar="number",
        type=float,
        help="Overwrites the learning rate",
    )
    parser.add_argument(
        "--checkpoint_path",
        dest="checkpoint_path",
        metavar="path",
        help="Overwrites the path to the saved checkpoint containing the model weights",
    )
    parser.add_argument(
        "--plot_history",
        dest="plot_history",
        metavar="boolean (default: false)",
        type=bool,
        help="Plots the model training history",
    )

    args = parser.parse_args()
    config_path = args.config
    config = Config(config_path)
    plot_history = False

    # Overwrite config
    if args.train_files_path:
        config.train.files_path = args.train_files_path
    if args.train_mask_files_path:
        config.train.mask_files_path = args.train_mask_files_path
    if args.test_file_path:
        config.test_file_path = args.test_file_path
    if args.test_threshold:
        config.test_threshold = args.test_threshold
    if args.epochs:
        config.train.epochs = args.epochs
    if args.loss:
        config.train.loss = args.loss
    if args.model:
        config.model = args.model
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.learning_rate:
        config.train.learning_rate = args.learning_rate
    if args.checkpoint_path:
        config.train.checkpoint_path = args.checkpoint_path

    if args.plot_history:
        plot_history = True

    # Set seed to get reproducable experiments
    seed_value = 33
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    # ToDo: Create model
    model = create_model(config)

    # ToDo: Train model
    model.train()
    history = model.history
    if not history == None:
        epochs = len(history.epoch) + model.initial_epoch
        model.model.save_weights(
            config.train.checkpoints_path + "/model-{0:04d}.ckpts".format(epochs)
        )

        if plot_history:
            model.plot_history()

    K.clear_session()


if __name__ == "__main__":
    main()
