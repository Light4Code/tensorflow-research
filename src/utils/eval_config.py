from .config_util import get_entry


class EvalConfig:
    """Evaluation configuration
    
    Returns:
        [type] -- Evaluation configuration
    """

    def __init__(self, root_node: dict):
        super().__init__()
        self.raw = root_node
        self._files_path = get_entry(self.raw, "files_path", None)
        self._threshold = get_entry(self.raw, "threshold", None)

    @property
    def files_path(self) -> str:
        """Gets or sets the path to the eval files"""
        return self._files_path

    @files_path.setter
    def files_path(self, value: str) -> None:
        self._files_path = value

    @property
    def threshold(self) -> float:
        """Gets or sets the threshold. Used for example for difference images"""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value
