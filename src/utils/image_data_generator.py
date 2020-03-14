import tensorflow as tf


def create_image_data_generator(image_generator_config):
    return tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=image_generator_config.featurewise_center,
        featurewise_std_normalization=image_generator_config.featurewise_std_normalization,
        rotation_range=image_generator_config.rotation_range,
        width_shift_range=image_generator_config.width_shift_range,
        horizontal_flip=image_generator_config.horizonal_flip,
        height_shift_range=image_generator_config.height_shift_range,
        zoom_range=image_generator_config.zoom_range,
        fill_mode="nearest",
    )
