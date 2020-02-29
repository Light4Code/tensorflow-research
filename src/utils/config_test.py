import os
import unittest
from utils.config import Config

class Test_Config(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)
        self.current_directory = os.getcwd()
        self.config_file_path = self.current_directory + '/config/sample.json'

    def test_init(self):
        config = Config(self.config_file_path)
        self.assertEqual(512, config.input_width)
        self.assertEqual(512, config.input_height)
        self.assertEqual(1, config.input_bpp)
        self.assertEqual('PATH_TO_TRAINING_FILES', config.train_files_path)

if __name__ == '__main__':
    unittest.main()
