import cv2
import glob
import numpy as np


class ImageUtil():
    def __init__(self):
        self.cv2_grayscale = 0
        self.cv2_color = 1
        self.cv2_unchanged = -1

    def load_images(self, images_path, color_mode=-1):
        png_files = glob.glob(images_path + '/*.png')
        images = []
        for png_path in png_files:
            images.append(self.load_image(png_path, color_mode))
        return images

    def load_image(self, image_path, color_mode=-1):
        return cv2.cvtColor(cv2.imread(image_path, color_mode), cv2.COLOR_BGR2RGB)

    def resize_image(self, image, target_width, target_height):
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    def get_color_mode(self, bpp):
        mode = self.cv2_grayscale
        if bpp == 3:
            mode = self.cv2_color
        elif bpp != 1 and bpp != 3:
            mode = self.cv2_unchanged
        return mode

    def normalize(self, image, shape):
        return image.reshape(shape) / 255.0  # create normalizes image

    def create_diff(self, original_image, predicted_image):
        diff = original_image - predicted_image
        return np.abs(diff)

    def apply_threshold(self, diff_image, threshold):
        diff_image[diff_image < threshold] = 0
        diff_image[diff_image >= threshold] = 1
        return diff_image
