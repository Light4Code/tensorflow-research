from utils.custom_types import Vector
from tensorflow.keras.models import Model


class BaseBackbone:
    @property
    def input_shape(self) -> Vector:
        return self._input_shape

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, value: Model) -> None:
        self._model = value

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def output_name(self) -> str:
        return self._output_name

    def __init__(
        self,
        input_shape: Vector,
        hidden_activation: str,
        output_activation: str,
        leaky_alpha: float = 0.1,
    ):
        self._input_shape = input_shape
        self._input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self._model = None
        self._input_name = "Placeholder"
        self._output_name = "output"

    def summary(self):
        return self.model.summary()

