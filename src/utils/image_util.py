import glob
import ntpath
import os
from typing import Type

import cv2
import numpy as np
from skimage.util import random_noise

from .custom_types import Vector

cv2_grayscale = 0
cv2_color = 1
cv2_unchanged = -1


def load_images_and_masks(root_path: str, target_shape: Vector) -> ([], [], [], []):
    class_dirs = []
    for di in sorted(glob.glob(root_path + "/*")):
        if os.path.isdir(di) and not str.endswith(di, "masks"):
            class_dirs.append(di)
    class_count = len(class_dirs)

    class_labels = []
    class_indexes = []
    train_x = []
    tmp_masks = []

    if class_count == 0:
        print("No classes detected, will continue without classes!")
        images, masks = _load_images_and_masks(root_path, target_shape)
        for idx in range(len(images)):
            train_x.append(
                np.array(images[idx], dtype=np.float32).reshape(target_shape)
            )
            if len(masks) > 0:
                tmp_masks.append(
                    np.array(masks[idx], dtype=np.float32).reshape(target_shape)
                )
    else:
        class_count = 0
        for class_dir in class_dirs:
            images, masks = _load_images_and_masks(class_dir, target_shape)
            for idx in range(len(images)):
                class_labels.append("{0}".format(class_count))
                class_indexes.append(class_count)
                train_x.append(
                    np.array(images[idx], dtype=np.float32).reshape(target_shape)
                )
            if len(masks) > 0:
                tmp_masks.append(
                    np.array(masks[idx], dtype=np.float32).reshape(target_shape)
                )
            class_count += 1

    return (
        np.array(class_labels),
        np.array(class_indexes, dtype=np.int),
        np.array(train_x, dtype=np.float32),
        np.array(tmp_masks, dtype=np.float32),
    )


def draw_mask(image, mask, shape: Vector):
    img = np.array(image * 255, dtype=np.uint8)
    mas = np.array(mask * 255, dtype=np.uint8)
    return np.array(cv2.bitwise_not(img, img, mask=mas)).reshape(shape)


def load_images(images_path: str, color_mode=-1) -> []:
    png_files = glob.glob(images_path + "/*.png")
    png_files = sorted(png_files)
    images = []
    for png_path in png_files:
        images.append(load_image(png_path, color_mode))
    return images


def save_image(image, path: str) -> None:
    img = image
    if image.shape[2] > 1:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def load_image(image_path: str, color_mode=-1):
    img = cv2.imread(image_path, color_mode)
    try:
        if color_mode == cv2_color or color_mode == cv2_unchanged:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            return img
    except:
        return None


def resize_image(image, target_width: int, target_height: int):
    return cv2.resize(
        image, (target_width, target_height), interpolation=cv2.INTER_CUBIC
    )


def normalize(image, shape: Vector):
    return np.array(
        image.reshape(shape) / 255.0, dtype=np.float32
    )  # create normalized image


def create_diff(original_image, predicted_image, threshold: float) -> ([], float):
    diff = cv2.subtract(original_image, predicted_image)
    se = np.sum(diff * diff)
    if threshold != 0:
        return np.array(diff > threshold, dtype=np.float32), float(se)
    else:
        return np.array(np.abs(diff), dtype=np.float32), float(se)


def create_noisy_images(original_images):
    lst_noisy = []
    sigma = 0.155
    for image in original_images:
        noisy = random_noise(image, var=sigma ** 2)
        lst_noisy.append(noisy)
    return np.array(lst_noisy)


def _get_color_mode(bpp: int):
    mode = cv2_grayscale
    if bpp == 3:
        mode = cv2_color
    elif bpp != 1 and bpp != 3:
        mode = cv2_unchanged
    return mode


def _load_images_and_masks(images_path: str, target_shape: Vector) -> ([], []):
    image_files = sorted(glob.glob(images_path + "/*.png"))
    images = []
    masks = []
    mode = _get_color_mode(target_shape[2])
    contains_masks = False
    image_dir = ntpath.dirname(image_files[0])
    if os.path.exists(image_dir + "/masks"):
        contains_masks = True
    for f in image_files:
        image = load_image(f, mode)
        resized_image = resize_image(image, target_shape[1], target_shape[0])
        shape = image.shape
        images.append(normalize(resized_image, target_shape))
        if contains_masks:
            mask = _load_mask(f, (shape[0], shape[1], 1))
            resized_mask = resize_image(mask, target_shape[1], target_shape[0])
            masks.append(normalize(resized_mask, target_shape))

    return (images, masks)


def _load_mask(image_path: str, shape: Vector) -> []:
    image_filename = ntpath.basename(image_path)
    image_dir = ntpath.dirname(image_path)
    mask_path = image_dir + "/masks/" + image_filename
    if os.path.exists(mask_path):
        mask = load_image(mask_path, color_mode=cv2_grayscale)
        mask = np.array(mask, dtype=np.uint8)
        mask = mask.reshape(shape)
    else:
        mask = np.zeros(shape, dtype=np.uint8)
    return mask
