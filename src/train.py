import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam

import backbones
import utils.plots as plots
from train_engine import TrainEngine
from utils import load_dataset
from utils.config import Config
from utils.environment_util import setup_environment
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

    setup_environment()

    input_shape = config.input_shape
    train_x, train_y, eval_x, eval_y = load_dataset(config.train.files_path, input_shape)

    backbone = backbones.AutoEncoderFullConnected(input_shape)
    optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0005)

    train_engine = TrainEngine(
        input_shape,
        backbone.model,
        optimizer,
        checkpoints_save_path=config.train.checkpoints_path,
        last_checkpoint_path=config.train.checkpoint_path,
    )
    loss, acc, val_loss, val_acc = train_engine.train(train_x, train_y, eval_x, eval_y, epochs=1000)
    if plot_history:
        plots.plot_history(loss, acc, val_loss, val_acc)
        predictions = train_engine.model.predict(np.array([eval_x[0]], dtype=np.float32), batch_size=1)
        plots.plot_prediction(predictions, [eval_x[0]], input_shape, config.eval.threshold)
    
    K.clear_session()


if __name__ == "__main__":
    main()
