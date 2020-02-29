import cv2
import glob


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
        return cv2.imread(image_path, color_mode)

    def resize_image(self, image, target_width, target_height):
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
