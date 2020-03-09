# root folder
# |-class folder
#     |-images
#     |-mask folder
#         |- mask images
import argparse
import os
from glob import glob

import numpy as np
import tensorflow as tf

from utils import *


def main():
    parser = argparse.ArgumentParser(
        description="Used to generate images, expects one subfolder for each class and if masks are used also a subfolder for the masks:\n"
        + "root folder\n"
        "  |-class folder\n"
        "      |-images\n"
        "      |-mask folder\n"
        "          |- mask images"
    )
    parser.add_argument(
        "config",
        metavar="config",
        help="Path to the configuration file containing all parameters for model training",
    )
    parser.add_argument(
        "--input_path",
        dest="input_path",
        metavar="path",
        help="Path to root images",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        metavar="path",
        help="Path to root images",
    )
    parser.add_argument(
        "--rounds",
        dest="rounds",
        metavar="number (default == 1",
        type=int,
        help="How many times the same folder should be generated",
    )


    args = parser.parse_args()
    config_path = args.config
    input_path = args.input_path
    output_path = args.output_path
    rounds = 1
    if args.rounds:
      rounds = args.rounds

    config = Config(config_path)
    image_util = ImageUtil()

    class_dirs = []
    for di in glob(input_path + "/*"):
      if os.path.isdir(di):
        class_dirs.append(di)
    class_count = len(class_dirs)
    image_dictionary = []
    if class_count == 0:
        print("No classes detected, will continue without classes!")
        images = image_util.load_images(input_path)
        masks = []
        masks_path = input_path + "/masks"
        if os.path.exists(masks_path):
            masks = image_util.load_images(masks_path)
        d = dict(index=0, class_name="0", images=images, masks=masks)
        image_dictionary.append(d)
    else:
        class_count = 0
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            images = image_util.load_images(class_dir)
            masks = []
            masks_path = class_dir + "/masks"
            if os.path.exists(masks_path):
                masks = image_util.load_images(masks_path)
            d = dict(index=class_count, class_name=class_name,
                     images=images, masks=masks)
            image_dictionary.append(d)
            class_count += 1
    
    for r in range(rounds):
      for d in image_dictionary:
        images, masks = generate_images(config, image_util, output_path, r, d["class_name"], d["images"], d["masks"])
        d["images"] = images
        d["masks"] = masks

def generate_images(config, image_util, output_path, prefix, class_name, images, masks):
    images_output_path = output_path + "/" + class_name
    masks_output_path = images_output_path + "/masks"

    if not os.path.exists(images_output_path):
      os.makedirs(images_output_path)
    if not os.path.exists(masks_output_path):
      os.makedirs(masks_output_path)

    resized_images = []
    resized_masks = []
    for image in images:
      resized_images.append(image_util.resize_image(image, config.input_shape[1], config.input_shape[0]))
    for mask in masks:
      resized_masks.append(image_util.resize_image(mask, config.input_shape[1], config.input_shape[0]))

    resized_images = np.array(resized_images)
    resized_masks = np.array(resized_masks)
    seed = 33
    new_images = []
    new_masks = []

    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=config.image_data_generator_featurewise_center,
        featurewise_std_normalization=config.image_data_generator_featurewise_std_normalization,
        rotation_range=config.image_data_generator_rotation_range,
        width_shift_range=config.image_data_generator_width_shift_range,
        horizontal_flip=config.image_data_generator_horizonal_flip,
        height_shift_range=config.image_data_generator_height_shift_range,
        zoom_range=config.image_data_generator_zoom_range,
        fill_mode='nearest',
    )

    image_data_generator.fit(resized_images, augment=True, seed=seed)

    image_data_flow = image_data_generator.flow(
        resized_images,
        batch_size=1,
        seed=seed,
        save_to_dir=images_output_path,
        save_prefix=prefix
    )

    for _ in range(image_data_flow.n):
      img = image_data_flow.next()
      new_images.append(img[0])

    if len(masks) > 0:
        mask_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=config.image_data_generator_featurewise_center,
            featurewise_std_normalization=config.image_data_generator_featurewise_std_normalization,
            rotation_range=config.image_data_generator_rotation_range,
            width_shift_range=config.image_data_generator_width_shift_range,
            horizontal_flip=config.image_data_generator_horizonal_flip,
            height_shift_range=config.image_data_generator_height_shift_range,
            zoom_range=config.image_data_generator_zoom_range,
            fill_mode='nearest',
        )
        mask_data_generator.fit(resized_masks, augment=True, seed=seed)
        mask_data_flow = mask_data_generator.flow(
            resized_masks,
            batch_size=1,
            seed=seed,
            save_to_dir=masks_output_path,
            save_prefix=prefix
        )
        for _ in range(mask_data_flow.n):
          ma = mask_data_flow.next()
          new_masks.append(ma[0])
    return (new_images, new_masks)


if __name__ == "__main__":
    main()
