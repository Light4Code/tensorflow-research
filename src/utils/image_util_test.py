import unittest
from utils.image_util import ImageUtil

class Test_ImageUtil(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)
        self.image_util = ImageUtil()

    def test_load_image_empty_path(self):
        image = self.image_util.load_image('')
        self.assertIsNone(image)
    
    def test_load_images_empty_path(self):
        images = self.image_util.load_images('')
        self.assertListEqual([], images)

if __name__ == '__main__':
    unittest.main()