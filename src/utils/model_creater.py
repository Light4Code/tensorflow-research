from models.anomaly_detection.advanced_model import AdvancedModel
from models.anomaly_detection.fast_model import FastModel
from models.segmentation.satellite_unet_model import SatelliteUnetModel
from models.segmentation.custom_unet_model import CustomUnetModel
from models.segmentation.small_unet_model import SmallUnetModel


def create_model(config):
    if config.model == "fast":
        model_container = FastModel(config)
    elif config.model == "advanced":
        model_container = AdvancedModel(config)
    elif config.model == "small_unet":
        model_container = SmallUnetModel(config)
    elif config.model == "satellite_unet":
        model_container = SatelliteUnetModel(config)
    elif config.model == "custom_unet":
        model_container = CustomUnetModel(config)
    else:
        TypeError

    return model_container
