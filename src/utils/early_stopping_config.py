from .config_util import get_entry


class EarlyStoppingConfig:
    """Early stopping configuration
    
    Returns:
        [type] -- Early stopping
    """

    def __init__(self, root_node: dict):
        super().__init__()
        self.raw = root_node
        self._val_loss_epochs = get_entry(self.raw, "val_loss_epochs", 0)

    @property
    def val_loss_epochs(self) -> int:
        """Gets or sets the validation loss epochs"""
        return self._val_loss_epochs

    @val_loss_epochs.setter
    def val_loss_epochs(self, value: int) -> None:
        self._val_loss_epochs = value
