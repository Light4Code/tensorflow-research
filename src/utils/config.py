import json


class Config():
    def __init__(self, config_path):
        super().__init__()
        with open(config_path) as json_file:
            data = json.load(json_file)
            self.input_shape = data['input_shape']

            train = data['train']
            self.model = train['model']
            self.train_files_path = train['files_path']
            self.batch_size = train['batch_size']
            self.epochs = train['epochs']
            self.learning_rate = train['learning_rate']
            self.loss = train['loss']
            self.optimizer = train['optimizer']

            try:
                self.image_data_generator = train['image_data_generator']
                self.image_data_generator_horizonal_flip = self.image_data_generator['horizontal_flip']
                self.image_data_generator_zoom_range = self.image_data_generator['zoom_range']
                self.image_data_generator_width_shift_range = self.image_data_generator['width_shift_range']
                self.image_data_generator_height_shift_range = self.image_data_generator['height_shift_range']
                self.image_data_generator_rotation_range = self.image_data_generator['rotation_range']
            except:
                self.image_data_generator = None
                self.image_data_generator_horizonal_flip = None
                self.image_data_generator_zoom_range = None
                self.image_data_generator_width_shift_range = None
                self.image_data_generator_height_shift_range = None
                self.image_data_generator_rotation_range = None

            try:
                test = data['test']
                self.test_file_path = test['file_path']
                self.test_threshold = test['threshold']
            except:
                self.test_file_path = None
                self.test_threshold = None
