import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import utils.image_util as iu
from utils.custom_types import Vector


def plot_history(loss, acc, val_loss, val_acc):
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.title("Loss")
    plt.grid()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Test"], loc="upper left")

    plt.subplot(2, 1, 2)
    plt.title("Accuracy")
    plt.grid()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()


def plot_difference(
    predictions, test_images, input_shape: Vector, threshold: float = 0.0
):
    plt.figure(figsize=(20, 10))
    pred_count = len(predictions)
    plt_shape = (input_shape[0], input_shape[1])
    plt_cmap = "gray"
    if input_shape[2] > 1:
        plt_shape = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
    index = 1
    plt_index = 0
    for test_image in test_images:
        original_image = test_image.reshape(plt_shape)
        pred_image = predictions[plt_index].reshape(plt_shape)
        diff, se = iu.create_diff(original_image, pred_image, threshold)
        mask = ma.masked_where(diff == False, diff)
        plt.subplot(pred_count, 4, index)
        plt.title("Original")
        plt.imshow(original_image, interpolation="none", cmap=plt_cmap)
        index += 1
        plt.subplot(pred_count, 4, index)
        plt.title("Prediction")
        plt.imshow(pred_image, interpolation="none", cmap=plt_cmap)
        index += 1
        plt.subplot(pred_count, 4, index)
        plt.title("Diff (SE: {0})".format(round(se, 2)))
        plt.imshow(diff, interpolation="none", cmap=plt_cmap)
        index += 1
        plt.subplot(pred_count, 4, index)
        plt.title("Overlay")
        plt.imshow(original_image, interpolation="none", cmap=plt_cmap)
        plt.imshow(mask, cmap="jet", interpolation="none", alpha=0.7)
        index += 1
        plt_index += 1
    plt.show()


def plot_prediction(
    predictions, test_images, input_shape: Vector, threshold: float = 0.4
):
    plt.figure(figsize=(20, 10))
    pred_count = len(predictions)
    plt_shape = (input_shape[0], input_shape[1])
    plt_cmap = "gray"
    if input_shape[2] > 1:
        plt_shape = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
    index = 1
    plt_index = 0
    for test_image in test_images:
        original_image = test_image.reshape(plt_shape)
        pred_image = predictions[plt_index].reshape(plt_shape)
        mask = ma.masked_where(pred_image < threshold, pred_image)
        plt.subplot(pred_count, 3, index)
        plt.title("Original")
        plt.imshow(original_image, interpolation="none", cmap=plt_cmap)
        index += 1
        plt.subplot(pred_count, 3, index)
        plt.title("Prediction")
        plt.imshow(pred_image, interpolation="none", cmap=plt_cmap)
        index += 1
        plt.subplot(pred_count, 3, index)
        plt.title("Overlay")
        plt.imshow(original_image, interpolation="none", cmap=plt_cmap)
        plt.imshow(mask, cmap="jet", interpolation="none", alpha=0.7)
        index += 1
        plt_index += 1
    plt.show()


def plot_classification(predictions, test_images, input_shape: Vector, classes):
    plt.figure(figsize=(20, 10))
    pred_count = len(predictions)
    plt_shape = (input_shape[0], input_shape[1])
    plt_cmap = "gray"
    if input_shape[2] > 1:
        plt_shape = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
    index = 1
    plt_index = 0
    for test_image in test_images:
        original_image = test_image.reshape(plt_shape)
        pred = predictions[plt_index]
        c_idx = np.argmax(pred)
        plt.subplot(pred_count, 1, index)
        plt.title("{0} ({1})".format(classes[c_idx], pred[0][c_idx]))
        plt.imshow(original_image, interpolation="none", cmap=plt_cmap)
        index += 1
        plt_index += 1
    plt.show()
