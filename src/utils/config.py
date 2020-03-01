import json
class Config():
    def __init__(self, config_path):
        super().__init__()
        with open(config_path) as json_file:
            data = json.load(json_file)
            self.input_shape = data['input_shape']

            train = data['train']
            self.train_files_path = train['files_path']
            self.batch_size = train['batch_size']
            self.epochs = train['epochs']
            self.learning_rate = train['learning_rate']

            try:
                test = data['test']
                self.test_file_path = test['file_path']
                self.test_threshold = test['threshold']
            except:
                self.test_file_path = None
                self.test_threshold = None

