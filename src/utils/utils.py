import random

import numpy as np
from sklearn.model_selection import train_test_split

import utils.image_util as iu
from utils.custom_types import Vector


def load_dataset(root_data_path:str, target_shape: Vector, shuffle:bool=True, validation_split:float=0.2):
  (class_labels, class_indexes, tmp_train_x, tmp_masks) = iu.load_images_and_masks(root_data_path, target_shape)

  tmp_train_y = None
  if len(tmp_masks) > 0:
    tmp_train_y = tmp_masks
    print("Using masks as train_y")
  elif len(class_indexes) > 0:
    tmp_train_y = class_indexes
    print("Using class indexes as train_y")
  else:
    tmp_train_y = tmp_train_x
    print("Using train_x as train_y")

  if shuffle:
    zipped = list(zip(tmp_train_x, tmp_train_y))
    random.shuffle(zipped)
    tmp_train_x, tmp_train_y = zip(*zipped)

  if validation_split <= 0:
    return (np.array(tmp_train_x, dtype=np.float32), np.array(tmp_train_y, dtype=np.float32), None, None)
  else:
    tmp_train_x, tmp_test_x, tmp_train_y, tmp_test_y = train_test_split(tmp_train_x, tmp_train_y, test_size=validation_split)
    print("Split dataset into {0} train and {1} test data".format(len(tmp_train_x), len(tmp_test_x)))
    return (np.array(tmp_train_x, dtype=np.float32), np.array(tmp_train_y, dtype=np.float32), np.array(tmp_test_x, dtype=np.float32), np.array(tmp_test_y, dtype=np.float32))

def print_epoch_statistics(hist_loss, hist_acc, hist_val_loss, hist_val_acc, executed_epochs: int, epochs: int, initial_epoch:int=0):
  print("Epoch {0}/{1}\tloss: {2}\tacc: {3}\tval_loss: {4}\tval_acc: {5}".format(executed_epochs + initial_epoch, 
              epochs + initial_epoch, 
              round(np.average(hist_loss), 5),
              round(float(np.average(hist_acc)), 5),
              round(float(np.average(hist_val_loss)), 5),
              round(float(np.average(hist_val_acc)), 5)))
