from .config_util import get_entry


class ImageGeneratorConfig:
    def __init__(self):
        super().__init__()
        self._loop_count = 1
        self._horizonal_flip = False
        self._zoom_range = 0.0
        self._width_shift_range = 0.0
        self._height_shift_range = 0.0
        self._rotation_range = 0.0
        self._featurewise_center = False
        self._featurewise_std_normalization = False

    @property
    def loop_count(self) -> int:
        """Gets or sets the loop count.
        The count how many times the image generator will be executed."""
        return self._loop_count

    @loop_count.setter
    def loop_count(self, value: int) -> None:
        self._loop_count = value

    @property
    def horizontal_flip(self) -> bool:
        """Gets or sets a value indicating whether horizontal flip should be used"""
        return self._horizonal_flip

    @horizontal_flip.setter
    def horizontal_flip(self, value: bool) -> None:
        self._horizonal_flip = value

    @property
    def zoom_range(self) -> float:
        """Gets or sets the zoom range"""
        return self._zoom_range

    @zoom_range.setter
    def zoom_range(self, value: float) -> None:
        self._zoom_range = value

    @property
    def width_shift_range(self) -> float:
        """Gets or sets the width shift range"""
        return self._width_shift_range

    @width_shift_range.setter
    def width_shift_range(self, value: float) -> None:
        self._width_shift_range = value

    @property
    def height_shift_range(self) -> float:
        """Gets or sets the height shift range"""
        return self._height_shift_range

    @height_shift_range.setter
    def height_shift_range(self, value: float) -> None:
        self._height_shift_range = value

    @property
    def rotation_range(self) -> float:
        """Gets or sets the rotation range"""
        return self._rotation_range

    @rotation_range.setter
    def rotation_range(self, value: float) -> None:
        self._rotation_range = value

    @property
    def featurewise_center(self) -> bool:
        """Gets or sets a value indicating whether featurewise center should be used"""
        return self._featurewise_center

    @featurewise_center.setter
    def featurewise_center(self, value: bool) -> None:
        self._featurewise_center = value

    @property
    def featurewise_std_normalization(self) -> bool:
        """Gets or sets a value indicating whether featurewise std normalization should be used"""
        return self._featurewise_std_normalization

    @featurewise_std_normalization.setter
    def featurewise_std_normalization(self, value: bool) -> None:
        self._featurewise_std_normalization = value
