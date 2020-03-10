import matplotlib.pyplot as plt
import numpy.ma as ma

from utils import ImageUtil


def plot_difference(config, predictions, test_images):
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
      diff = image_util.create_diff(original_image, pred_image)
      mask = ma.masked_where(diff < config.test_threshold, diff)
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
      mask = ma.masked_where(pred_image < config.test_threshold, pred_image)
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
