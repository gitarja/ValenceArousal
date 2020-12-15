import unittest
from KnowledgeDistillation.Utils.DataGenerator import DataGenerator
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetch
from Conf.Settings import DATASET_PATH, FEATURES_N, ECG_RAW_N

class MyTestCase(unittest.TestCase):

    # def test_dimension(self):
    #     path = "D:\\usr\\pras\\data\\EmotionTestVR\\"
    #     training_file = "D:\\usr\\pras\\data\\EmotionTestVR\\training.csv"
    #     testing_file = "D:\\usr\\pras\\data\\EmotionTestVR\\testing.csv"
    #     ecg_length = 11000
    #     generator = DataGenerator(path=path, training_list_file=training_file,
    #                        testing_list_file=testing_file, ecg_length=ecg_length)
    #     inputs = generator.fetchData()
    #     self.assertEqual(4, len(inputs))
    #     self.assertEqual(ecg_length, inputs[1].shape[1])


    def test_generatorFeatures(self):
        training_data = DATASET_PATH + "training_data_1.csv"
        testing_data = DATASET_PATH + "test_data_1.csv"
        validation_data = DATASET_PATH + "validation_data_1.csv"

        generator = DataFetch(train_file=training_data, test_file=testing_data, validation_file=validation_data, ECG_N=ECG_RAW_N,max_scaler=None, norm_scaler=None)
        X, y_ar, y_val = generator.fetch(training_mode=0, KD=False)

        self.assertEqual(X.shape, (2480, ))
        # self.assertEqual(ecg.shape, (ECG_RAW_N, ))





if __name__ == '__main__':
    unittest.main()
