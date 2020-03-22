import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from utils import ImageUtil


def plot_difference(config, predictions, test_images):
    plt.figure(figsize=(20, 10))
    image_util = ImageUtil()
    pred_count = len(predictions)
    plt_shape = (config.input_shape[0], config.input_shape[1])
    plt_cmap = "gray"
    if config.input_shape[2] > 1:
        plt_shape = (
            config.input_shape[0],
            config.input_shape[1],
            config.input_shape[2],
        )
    index = 1
    plt_index = 0
    for test_image in test_images:
        original_image = test_image.reshape(plt_shape)
        pred_image = predictions[plt_index].reshape(plt_shape)
        diff = image_util.create_diff(original_image, pred_image, config.eval.threshold)
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
        plt.title("Difference")
        plt.imshow(diff, interpolation="none", cmap=plt_cmap)
        index += 1
        plt.subplot(pred_count, 4, index)
        plt.title("Overlay")
        plt.imshow(original_image, interpolation="none", cmap=plt_cmap)
        plt.imshow(mask, cmap="jet", interpolation="none", alpha=0.7)
        index += 1
        plt_index += 1
    plt.show()


def plot_prediction(config, predictions, test_images):
    plt.figure(figsize=(20, 10))
    pred_count = len(predictions)
    plt_shape = (config.input_shape[0], config.input_shape[1])
    plt_cmap = "gray"
    if config.input_shape[2] > 1:
        plt_shape = (
            config.input_shape[0],
            config.input_shape[1],
            config.input_shape[2],
        )
    index = 1
    plt_index = 0
    for test_image in test_images:
        original_image = test_image.reshape(plt_shape)
        pred_image = predictions[plt_index].reshape(plt_shape)
        mask = ma.masked_where(pred_image < config.eval.threshold, pred_image)
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


def plot_classification(config, predictions, test_images, classes):
    plt.figure(figsize=(20, 10))
    pred_count = len(predictions)
    plt_shape = (config.input_shape[0], config.input_shape[1])
    plt_cmap = "gray"
    if config.input_shape[2] > 1:
        plt_shape = (
            config.input_shape[0],
            config.input_shape[1],
            config.input_shape[2],
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
