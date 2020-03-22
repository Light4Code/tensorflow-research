from typing import Type
from .config_util import get_entry
from .image_generator_config import ImageGeneratorConfig
from .early_stopping_config import EarlyStoppingConfig


class TrainConfig:
    """Configuration used for training
    
    Raises:
        ValueError: Inavlid configuration
    
    Returns:
        [type] -- Train configuration
    """

    def __init__(self, root_node: dict):
        super().__init__()
        self.raw = root_node
        self._files_path = get_entry(root_node, "files_path", None)
        self._alternative_files_path = get_entry(
            root_node, "alternative_files_path", None
        )

        self._epochs = get_entry(root_node, "epochs", 10)
        self._batch_size = get_entry(root_node, "batch_size", 1)
        self._learning_rate = get_entry(root_node, "learning_rate", 1e-4)
        self._loss = get_entry(root_node, "loss", None)
        self._decay = get_entry(root_node, "decay", 0)
        self._momentum = get_entry(root_node, "momentum", 0)
        self._optimizer = get_entry(root_node, "optimizer", None)
        self._validation_split = get_entry(root_node, "validation_split", 0.2)
        self._mask_files_path = get_entry(root_node, "mask_files_path", None)
        self._checkpoint_save_best_only = get_entry(
            root_node, "checkpoint_save_best_onky", False
        )
        self._checkpoint_path = get_entry(root_node, "checkpoint_path", None)
        self._checkpoints_path = get_entry(root_node, "checkpoints_path", None)
        self._checkpoint_save_period = get_entry(
            root_node, "checkpoint_save_period", 10
        )

        if not self._files_path:
            raise ValueError("Configuration needs the path to the training files!")

        try:
            self._image_data_generator = ImageGeneratorConfig(
                root_node["image_data_generator"]
            )
        except:
            self._image_data_generator = None

        try:
            self._early_stopping = EarlyStoppingConfig(root_node["early_stopping"])
        except:
            self._early_stopping = None

    @property
    def files_path(self) -> str:
        """Gets or sets the path to the train files"""
        return self._files_path

    @files_path.setter
    def files_path(self, value: str) -> None:
        self._files_path = value

    @property
    def alternative_files_path(self) -> str:
        """Gets or sets the alternative path to the train files. Used for example fake/NOK images."""
        return self._alternative_files_path

    @alternative_files_path.setter
    def alternative_files_path(self, value: str) -> None:
        self._alternative_files_path = value

    @property
    def epochs(self) -> int:
        """Gets or sets the epochs"""
        return self._epochs

    @epochs.setter
    def epochs(self, value: int) -> None:
        self._epochs = value

    @property
    def batch_size(self) -> int:
        """Gets or sets the batch size"""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self._batch_size = value

    @property
    def learning_rate(self) -> float:
        """Gets or sets the learning rate"""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self._learning_rate = value

    @property
    def loss(self) -> str:
        """Gets or sets the loss"""
        return self._loss

    @loss.setter
    def loss(self, value: str) -> None:
        self._loss = value

    @property
    def decay(self) -> float:
        """Gets or sets the decay"""
        return self._decay

    @decay.setter
    def decay(self, value: float) -> None:
        self._decay = value

    @property
    def momentum(self) -> float:
        """Gets or sets the momentum"""
        return self._momentum

    @momentum.setter
    def momentum(self, value: float) -> None:
        self._momentum = value

    @property
    def optimizer(self) -> str:
        """Gets or sets the optimizer"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: str) -> None:
        self._optimizer = value

    @property
    def validation_split(self) -> float:
        """Gets or sets the validation split"""
        return self._validation_split

    @validation_split.setter
    def validation_split(self, value: float) -> None:
        self._validation_split = value

    @property
    def mask_files_path(self) -> str:
        """Gets or sets the path to the mask files"""
        return self._mask_files_path

    @mask_files_path.setter
    def mask_files_path(self, value: str) -> None:
        self._mask_files_path = value

    @property
    def checkpoint_save_best_only(self) -> bool:
        """Gets or sets a value indicating only best checkpoint should be saved"""
        return self._checkpoint_save_best_only

    @checkpoint_save_best_only.setter
    def checkpoint_save_best_only(self, value: bool) -> None:
        self._checkpoint_save_best_only = value

    @property
    def checkpoint_path(self) -> str:
        """Gets or sets path to the checkpoint"""
        return self._checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, value: str) -> None:
        self._checkpoint_path

    @property
    def checkpoints_path(self) -> str:
        """Gets or sets the path where the checkpoints should be saved"""
        return self._checkpoints_path

    @checkpoints_path.setter
    def checkpoints_path(self, value: str) -> None:
        self._checkpoints_path

    @property
    def checkpoint_save_period(self) -> int:
        """Gets or sets the checkpoint save period"""
        return self._checkpoint_save_period

    @checkpoint_save_period.setter
    def checkpoint_save_period(self, value: int) -> None:
        self._checkpoint_save_period

    @property
    def image_data_generator(self) -> Type[ImageGeneratorConfig]:
        """Gets the image data generator config
        
        Returns:
            Type[ImageGeneratorConfig] -- Image data generator config
        """
        return self._image_data_generator

    @property
    def early_stopping(self) -> Type[EarlyStoppingConfig]:
        """Gets the early stopping config
        
        Returns:
            Type[EarlyStoppingConfig] -- Early stopping config
        """
        return self._early_stopping
