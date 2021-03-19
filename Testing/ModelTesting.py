import unittest
from KnowledgeDistillation.Models.EnsembleDistillModel import EnsembleStudentOneDim
from Conf.Settings import ECG_RAW_N
import tensorflow as tf
class MyTestCase(unittest.TestCase):
    def test_something(self):
        X = tf.random.uniform(shape=(1, ECG_RAW_N, 1))
        model = EnsembleStudentOneDim(num_output=12, classification=True)
        em_logit, ar_logit, val_logit = model(X)
        print(em_logit)
        self.assertEqual(em_logit, None)


if __name__ == '__main__':
    unittest.main()
