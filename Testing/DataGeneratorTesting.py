import unittest
from Libs.DataGenerator import DataGenerator

class MyTestCase(unittest.TestCase):
    def test_dimension(self):
        path = "D:\\usr\\pras\\data\\EmotionTestVR\\"
        training_file = "D:\\usr\\pras\\data\\EmotionTestVR\\training.csv"
        testing_file = "D:\\usr\\pras\\data\\EmotionTestVR\\testing.csv"
        ecg_length = 11000
        generator = DataGenerator(path=path, training_list_file=training_file,
                           testing_list_file=testing_file, ecg_length=ecg_length)
        inputs = generator.fetchData()
        self.assertEqual(4, len(inputs))
        self.assertEqual(ecg_length, inputs[1].shape[1])


if __name__ == '__main__':
    unittest.main()
