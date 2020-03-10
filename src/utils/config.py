import json


def get_entry(root_node, entry_name, default_value):
    node = default_value
    try:
        node = root_node[entry_name]
    except:
        pass
    return node

class ImageGeneratorConfig:
    def __init__(self, root_node):
        super().__init__()
        self.horizonal_flip = get_entry(root_node, "horizontal_flip", False)
        self.zoom_range = get_entry(root_node, "zoom_range", 0)
        self.width_shift_range = get_entry(root_node, "width_shift_range", 0)
        self.height_shift_range = get_entry(root_node, "height_shift_range", 0)
        self.rotation_range = get_entry(root_node, "rotation_range", 0)
        self.featurewise_center = get_entry(root_node, "featurewise_center", False)
        self.featurewise_std_normalization = get_entry(root_node, "featurewise_std_normalization", False)

class TrainConfig:
    def __init__(self, root_node):
        super().__init__()
        self.files_path = get_entry(root_node, "files_path", None)

        self.epochs = get_entry(root_node, "epochs", 10)
        self.batch_size = get_entry(root_node, "batch_size", 1)
        self.learning_rate = get_entry(root_node, "learning_rate", 1e-4)
        self.loss = get_entry(root_node, "loss", None)
        self.optimizer = get_entry(root_node, "optimizer", None)
        self.validation_split = get_entry(root_node, "validation_split", 0.2)
        self.mask_files_path = get_entry(root_node, "mask_files_path", None)
        self.checkpoint_save_best_only = get_entry(root_node, "checkpoint_save_best_onky", False)
        self.checkpoint_path = get_entry(root_node, "checkpoint_path", None)
        self.checkpoints_path = get_entry(root_node, "checkpoints_path", None)
        self.checkpoint_save_period = get_entry(root_node, "checkpoint_save_period", 10)
        
        if not self.files_path:
            raise ValueError("Configuration needs the path to the training files!")

        try:
            self.image_data_generator = ImageGeneratorConfig(root_node["image_data_generator"])
        except:
            self.image_data_generator = None

class Config:
    def __init__(self, config_path):
        super().__init__()
        with open(config_path) as json_file:
            data = json.load(json_file)
            self.input_shape = data["input_shape"]
            self.model = get_entry(data, "model", None)

            if not self.model:
                raise ValueError("Configuration needs the model name!")

            self.train = data["train"]
            self.test_files_path = None
            self.test_threshold = None

            try:
                test = data["test"]
                self.test_files_path = test["files_path"]
                try:
                    self.test_threshold = test["threshold"]
                except:
                    pass
            except:
                pass

            try:
                self.train = TrainConfig(data["train"])
            except:
                raise ValueError("Could not create train configuration!")

            try:
                self.image_data_generator = ImageGeneratorConfig(data["image_data_generator"])
            except:
                self.image_data_generator = None
