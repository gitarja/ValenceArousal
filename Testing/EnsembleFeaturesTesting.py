import unittest
from KnowledgeDistillation.Models.EnsembleFeaturesModel import EnsembleSeparateModel
import tensorflow as tf
class MyTestCase(unittest.TestCase):
    def test_something(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        N = 20
        X = tf.random.uniform(shape=(N, 100))
        y = tf.cast(tf.transpose(tf.random.categorical(tf.math.log([[0.5, 0.5]]), N)), tf.dtypes.float32)
        model = EnsembleSeparateModel(num_output=1)

        losses = model.trainSMCL(X, y, 1)
        self.assertEqual(losses.shape, (N, 6))


if __name__ == '__main__':
    unittest.main()
