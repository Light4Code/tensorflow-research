from typing import Type
from .config import Config
from models.anomaly_detection import *
from models.classification import *
from models.segmentation import *


def create_model(config: Type[Config]):
    """Create a new model by configuration definition
    
    Arguments:
        config {[type]} -- Main configuration
    
    Returns:
        [type] -- New created model
    """
    if config.model == "deep_autoencoder":
        model_container = DeepAutoencoderModel(config)
    elif config.model == "custom_conv_autoencoder":
        model_container = CustomConvAutoencoderModel(config)
    elif config.model == "conv_autoencoder":
        model_container = ConvAutoencoderModel(config)
    elif config.model == "small_unet":
        model_container = SmallUnetModel(config)
    elif config.model == "satellite_unet":
        model_container = SatelliteUnetModel(config)
    elif config.model == "custom_unet":
        model_container = CustomUnetModel(config)
    elif config.model == "vanilla_unet":
        model_container = VanillaUnetModel(config)
    elif config.model == "conv_classification":
        model_container = ConvClassificationModel(config)
    elif config.model == "adversarially_classification":
        model_container = AdversariallyClassificationModel(config)
    else:
        TypeError

    return model_container
