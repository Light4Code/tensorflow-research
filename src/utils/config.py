from typing import Type
import json
from .custom_types import Vector
from .config_util import get_entry
from .train_config import TrainConfig
from .eval_config import EvalConfig
from .image_generator_config import ImageGeneratorConfig


class Config:
    """Main configuration
    
    Raises:
        ValueError: Configuration needs a model name
        ValueError: Configuration needs at least a train node
    
    Returns:
        [type] -- New configuration
    """

    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path) as json_file:
            data = json.load(json_file)
            self._input_shape = data["input_shape"]
            self._model = get_entry(data, "model", None)

            if not self._model:
                raise ValueError("Configuration needs the model name!")

            self._train = data["train"]

            try:
                self._eval = EvalConfig(data["eval"])
            except:
                self._eval = EvalConfig(None)

            try:
                self._train = TrainConfig(data["train"])
            except:
                raise ValueError("Could not create train configuration!")

            try:
                self._image_data_generator = ImageGeneratorConfig(
                    data["image_data_generator"]
                )
            except:
                self._image_data_generator = None

    @property
    def model(self) -> str:
        """Gets or sets the model"""
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value

    @property
    def input_shape(self) -> Vector:
        """Gets or sets the input shape"""
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value: Vector):
        self._input_shape = value

    @property
    def train(self) -> Type[TrainConfig]:
        """Gets the train configuration
        
        Returns:
            Type[TrainConfig] -- Train configuration
        """
        return self._train

    @property
    def eval(self) -> Type[EvalConfig]:
        """Gets the evaluation configuration
        
        Returns:
            Type[EvalConfig] -- Evaluation configuration
        """
        return self._eval

    @property
    def image_data_generator(self) -> Type[ImageGeneratorConfig]:
        """Gets the image data generator configuration
        
        Returns:
            Type[ImageGeneratorConfig] -- Image data generator configuration
        """
        return self._image_data_generator
