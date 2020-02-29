import json
class Config():
    def __init__(self, config_path):
        super().__init__()
        with open(config_path) as json_file:
            data = json.load(json_file)
            input_shape = data['input_shape']
            self.input_width = input_shape['width']
            self.input_height = input_shape['height']
            self.input_bpp = input_shape['bpp']

            train = data['train']
            self.train_files_path = train['files_path']
