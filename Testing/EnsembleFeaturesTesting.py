import unittest
from KnowledgeDistillation.Models.EnsembleFeaturesModel import EnsembleSeparateModel
import tensorflow as tf
from KnowledgeDistillation.Models.EnsembleDistillModel import BaseStudentOneDim
from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N
class MyTestCase(unittest.TestCase):
    # def test_ensemble(self):
    #     physical_devices = tf.config.list_physical_devices('GPU')
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #     N = 20
    #     X = tf.random.uniform(shape=(N, 100))
    #     y = tf.cast(tf.transpose(tf.random.categorical(tf.math.log([[0.5, 0.5]]), N)), tf.dtypes.float32)
    #     model = EnsembleSeparateModel(num_output=3)
    #
    #     losses = model.trainSMCL(X, y, y, 1, 100)
    #     self.assertEqual(losses.shape, (N, 6))


    # def test_distill(self):
    #     dim = 96
    #     model = EnsembleStudent(num_output=1, ecg_size=(dim, dim))
    #     X = tf.random.uniform(shape=(1, dim, dim, 1))
    #     _, _, Xr = model(X)
    #     # model.summary()
    #
    #     self.assertEqual(X.shape, Xr.shape)

    def test_distill(self):
        dim = 11664
        model = BaseStudentOneDim()
        X = tf.random.uniform(shape=(1, dim, 1))
        Xr, _ = model(X)
        # model.summary()

        self.assertEqual(X.shape, Xr.shape)

if __name__ == '__main__':
    unittest.main()
