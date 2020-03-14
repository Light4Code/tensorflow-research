# root folder
# |-class folder
#     |-images
#     |-mask folder
#         |- mask images
import argparse
import os
from glob import glob

import cv2
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
        "--input_path", dest="input_path", metavar="path", help="Path to root images",
    )
    parser.add_argument(
        "--output_path", dest="output_path", metavar="path", help="Path to root images",
    )
    parser.add_argument(
        "--rounds",
        dest="rounds",
        metavar="number (default == 1",
        type=int,
        help="How many times the same folder should be generated",
    )
    parser.add_argument(
        "--original_image_path",
        dest="original_image_path",
        metavar="path",
        help="Path to the original image. If set, difference images will be created.",
    )

    args = parser.parse_args()
    config_path = args.config
    input_path = args.input_path
    output_path = args.output_path
    original_image_path = None
    rounds = 1
    if args.rounds:
        rounds = args.rounds
    if args.original_image_path:
        original_image_path = args.original_image_path

    config = Config(config_path)
    image_util = ImageUtil()

    original_image = None
    use_original_image = False
    if original_image_path:
        original_image = image_util.load_image(original_image_path)
        original_image = image_util.resize_image(
            original_image, config.input_shape[1], config.input_shape[0]
        )
        use_original_image = True

    class_dirs = []
    for di in glob(input_path + "/*"):
        if os.path.isdir(di):
            class_dirs.append(di)
    class_count = len(class_dirs)
    image_dictionary = []
    if class_count == 0:
        print("No classes detected, will continue without classes!")
        images, masks = image_util.load_images_and_masks(input_path)
        d = dict(index=0, class_name="0", images=images, masks=masks)
        image_dictionary.append(d)
    else:
        class_count = 0
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            images, masks = image_util.load_images_and_masks(class_dir)
            d = dict(
                index=class_count, class_name=class_name, images=images, masks=masks
            )
            image_dictionary.append(d)
            class_count += 1

    image_data_generator = create_image_data_generator(config)
    mask_data_generator = create_image_data_generator(config)

    for r in range(rounds):
        for d in image_dictionary:
            class_name = d["class_name"]
            create_dirs(output_path, class_name)
            images, masks = generate_images(
                image_data_generator,
                mask_data_generator,
                config,
                image_util,
                d["images"],
                d["masks"],
            )
            save_files(images, masks, r, output_path, class_name)
            if use_original_image:
                create_difference_images(
                    original_image, images, output_path + "/diff/" + class_name, r
                )
        print("Finished round (" + str(r + 1) + "/" + str(rounds) + ")")


import uuid


def create_image_data_generator(config):
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=config.image_data_generator.featurewise_center,
        featurewise_std_normalization=config.image_data_generator.featurewise_std_normalization,
        rotation_range=config.image_data_generator.rotation_range,
        width_shift_range=config.image_data_generator.width_shift_range,
        horizontal_flip=config.image_data_generator.horizonal_flip,
        height_shift_range=config.image_data_generator.height_shift_range,
        zoom_range=config.image_data_generator.zoom_range,
        fill_mode="nearest",
    )
    return data_generator


def save_files(images, masks, round_index, output_path, class_name):
    image_util = ImageUtil()
    for index in range(len(images)):
        uid = uuid.uuid4()
        img = images[index]
        mask = masks[index]
        image_util.save_image(
            img, "{0}/{1}/{2}_{3}.png".format(output_path, class_name, round_index, uid)
        )
        image_util.save_image(
            mask,
            "{0}/{1}/masks/{2}_{3}.png".format(
                output_path, class_name, round_index, uid
            ),
        )


def create_dirs(output_path, class_name):
    images_output_path = output_path + "/" + class_name
    masks_output_path = images_output_path + "/masks"

    if not os.path.exists(images_output_path):
        os.makedirs(images_output_path)
    if not os.path.exists(masks_output_path):
        os.makedirs(masks_output_path)


def generate_images(
    image_data_generator, mask_data_generator, config, image_util, images, masks,
):
    resized_images = []
    resized_masks = []
    for image in images:
        resized_images.append(
            image_util.resize_image(image, config.input_shape[1], config.input_shape[0])
        )
    for mask in masks:
        m = image_util.resize_image(mask, config.input_shape[1], config.input_shape[0])
        resized_masks.append(
            m.reshape((config.input_shape[1], config.input_shape[0], 1))
        )

    resized_images = np.array(resized_images)
    resized_masks = np.array(resized_masks)
    seed = 33
    new_images = []
    new_masks = []

    image_data_generator.fit(resized_images, augment=True, seed=seed)
    image_data_flow = image_data_generator.flow(resized_images, batch_size=1, seed=seed)

    for _ in range(image_data_flow.n):
        img = image_data_flow.next()
        new_images.append(np.array(img[0], dtype=np.uint8))

    if len(masks) > 0:
        mask_data_generator.fit(resized_masks, augment=True, seed=seed)
        mask_data_flow = mask_data_generator.flow(
            resized_masks, batch_size=1, seed=seed,
        )
        for _ in range(mask_data_flow.n):
            ma = mask_data_flow.next()
            new_masks.append(np.array(ma[0], dtype=np.uint8))

    return (new_images, new_masks)


def create_difference_images(original_image, images, output_path, round_index):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img_count = 0
    for img in images:
        diff = cv2.absdiff(original_image, img)
        cv2.imwrite(
            "{0}/diff_{1}_{2}.png".format(output_path, round_index, img_count), diff
        )
        img_count += 1


if __name__ == "__main__":
    main()
