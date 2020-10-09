import unittest
from Libs.DataGenerator import DataGenerator

class MyTestCase(unittest.TestCase):
    def test_dimension(self):
        teacher_path = "D:\\usr\\pras\\data\\EmotionTestVR\\Komiya\\results\\"
        student_path = "D:\\usr\\pras\\data\\EmotionTestVR\\Komiya\\results\\"
        feature_list_file = "D:\\usr\\pras\\data\\EmotionTestVR\\Komiya\\features_list.csv"
        generator = DataGenerator(student_features_path=student_path, teacher_features_path=teacher_path, features_list_file= feature_list_file)
        inputs = generator.fetchData()
        self.assertEqual(4, len(inputs))


if __name__ == '__main__':
    unittest.main()
