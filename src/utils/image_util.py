import glob
import os
import ntpath

import cv2
import numpy as np


class ImageUtil:
    def __init__(self):
        self.cv2_grayscale = 0
        self.cv2_color = 1
        self.cv2_unchanged = -1

    def load_images_and_masks(self, images_path, color_mode=-1):
        image_files = sorted(glob.glob(images_path + "/*.png"))
        images = []
        masks = []

        for f in image_files:
            image = self.load_image(f, color_mode)
            shape = image.shape
            images.append(image)
            masks.append(self.load_mask(f, (shape[0], shape[1], 1)))

        return (images, masks)

    def load_mask(self, image_path, shape):
        image_filename = ntpath.basename(image_path)
        image_dir = ntpath.dirname(image_path)
        mask_path = image_dir + "/masks/" + image_filename
        if os.path.exists(mask_path):
            mask = self.load_image(mask_path, color_mode=self.cv2_grayscale)
            mask = np.array(mask, dtype=np.uint8)
            mask = mask.reshape(shape)
        else:
            mask = np.zeros(shape, dtype=np.uint8)
        return mask

    def load_images(self, images_path, color_mode=-1):
        png_files = glob.glob(images_path + "/*.png")
        png_files = sorted(png_files)
        images = []
        for png_path in png_files:
            images.append(self.load_image(png_path, color_mode))
        return images

    def save_image(self, image, path):
        img = image
        if image.shape[2] > 1:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)

    def load_image(self, image_path, color_mode=-1):
        img = cv2.imread(image_path, color_mode)
        try:
            if color_mode == self.cv2_color or color_mode == self.cv2_unchanged:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                return img
        except:
            return None

    def resize_image(self, image, target_width, target_height):
        return cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_CUBIC
        )

    def get_color_mode(self, bpp):
        mode = self.cv2_grayscale
        if bpp == 3:
            mode = self.cv2_color
        elif bpp != 1 and bpp != 3:
            mode = self.cv2_unchanged
        return mode

    def normalize(self, image, shape):
        return image.reshape(shape) / 255.0  # create normalized image

    def create_diff(self, original_image, predicted_image, threshold):
        return cv2.subtract(original_image, predicted_image) > threshold

    def apply_threshold(self, diff_image, threshold):
        diff_image[diff_image < threshold] = 0
        diff_image[diff_image >= threshold] = 1
        return diff_image

    def create_empty_image(self, shape=(256, 256, 1)):
        return np.zeros(shape)

    def create_mask_images(self, config):
        train_images_path = config.train.files_path
        mask_images_path = config.train.mask_files_path
        train_file_names = glob.glob(train_images_path + "/*.png")
        original_mask_file_names = glob.glob(mask_images_path + "/*.png")
        mask_file_names = []

        train_file_names = sorted(train_file_names)
        original_mask_file_names = sorted(original_mask_file_names)

        for mask_file in original_mask_file_names:
            mask_file_names.append(ntpath.basename(mask_file))

        color_mode = self.get_color_mode(config.input_shape[2])
        input_shape = config.input_shape
        if color_mode == self.cv2_grayscale:
            input_shape = (input_shape[0], input_shape[1])
        masks = []
        mask_index = 0
        for train_file in train_file_names:
            tf = ntpath.basename(train_file)

            if tf not in mask_file_names:
                mask = self.create_empty_image(input_shape)
            else:
                mask = self.load_image(
                    original_mask_file_names[mask_index], color_mode,
                )
                mask = self.resize_image(mask, input_shape[1], input_shape[0])
                mask_index += 1
            masks.append(mask)
        return masks
