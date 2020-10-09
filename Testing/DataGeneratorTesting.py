import unittest
from Libs.DataGenerator import DataGenerator

class MyTestCase(unittest.TestCase):
    def test_dimension(self):
        teacher_path = "D:\\usr\\pras\\data\\EmotionTestVR\\Komiya\\results\\"
        student_path = "D:\\usr\\pras\\data\\EmotionTestVR\\Komiya\\results\\"
        feature_list_file = "D:\\usr\\pras\\data\\EmotionTestVR\\Komiya\\features_list.csv"
        ecg_length = 11000
        generator = DataGenerator(student_features_path=student_path, teacher_features_path=teacher_path, features_list_file= feature_list_file, ecg_length=ecg_length)
        inputs = generator.fetchData()
        self.assertEqual(4, len(inputs))
        self.assertEqual(ecg_length, inputs[1].shape[1])


if __name__ == '__main__':
    unittest.main()
