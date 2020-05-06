import os
import random
import numpy as np
import tensorflow as tf


def setup_environment(enable_gpu: bool = True):
    # Set seed to get reproducable experiments
    seed_value = 33
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    if enable_gpu:
        physical_devices = tf.config.list_physical_devices("GPU")
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass
