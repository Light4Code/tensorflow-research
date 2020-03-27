import argparse

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam

import backbones
import utils.plots as plots
from train_engine import TrainEngine
from utils import load_dataset, ImageGeneratorConfig, setup_environment


def main():
    parser = argparse.ArgumentParser(description="Used to train TensorFlow model")
    parser.add_argument(
        "train_files_path", metavar="path", help="Path to the training files.",
    )
    parser.add_argument(
        "test_files_path", metavar="path", help="Path to the test files",
    )
    parser.add_argument(
        "--plot_history",
        dest="plot_history",
        metavar="boolean (default: false)",
        type=bool,
        help="Plots the model training history",
    )

    plot_history = False
    args = parser.parse_args()
    if args.plot_history:
        plot_history = args.plot_history

    setup_environment()
    train_files_path = args.train_files_path
    eval_files_path = args.test_files_path

    input_shape = (240, 240, 1)
    generator_config = ImageGeneratorConfig()
    generator_config.loop_count = 10
    generator_config.horizontal_flip = True
    generator_config.zoom_range = 0.3
    generator_config.width_shift_range = 0.3
    generator_config.height_shift_range = 0.3
    generator_config.rotation_range = 10

    train_x, train_y, eval_x, eval_y = load_dataset(
        train_files_path, input_shape, validation_split=0
    )

    eval_x, eval_y, t1, t2, = load_dataset(
        eval_files_path, input_shape, validation_split=0
    )

    backbone = backbones.SegmentationVanillaUnet(input_shape)
    # optimizer = SGD(lr=0.0001, momentum=0.9, decay=0.0)
    optimizer = Adam(lr=0.00001)

    train_engine = TrainEngine(
        input_shape, backbone.model, optimizer, loss="binary_crossentropy"
    )

    loss, acc, val_loss, val_acc = train_engine.train(
        train_x,
        train_y,
        eval_x,
        eval_y,
        epochs=50,
        batch_size=10,
        image_generator_config=generator_config,
    )
    if plot_history:
        plots.plot_history(loss, acc, val_loss, val_acc)
        for idx in range(len(eval_x[:3])):
            predictions = train_engine.model.predict(
                np.array([eval_x[idx]], dtype=np.float32), batch_size=1
            )
            plots.plot_prediction(predictions, [eval_x[idx]], input_shape)

    K.clear_session()


if __name__ == "__main__":
    main()
