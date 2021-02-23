from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as K
import tensorflow as tf
#loss
class PCCLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='PCCLoss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        y_pred_mean = math_ops.reduce_mean(y_pred, keepdims=True)
        y_true_mean = math_ops.reduce_mean(y_true, keepdims=True)
        y_pred_m = y_pred - y_pred_mean
        y_true_m = y_true - y_true_mean
        y_pred_norm = tf.norm(y_pred_m, keepdims=True)
        y_true_norm = tf.norm(y_true_m, keepdims=True)

        pcc = (math_ops.reduce_sum(y_pred_m * y_true_m, keepdims=True)) / (
                math_ops.reduce_sum(y_pred_norm * y_true_norm, keepdims=True) + 1e-25)

        pcc = tf.maximum(tf.minimum(pcc, 1.0), -1.0)

        return pcc

class CCCLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='CCCLoss'):
        super().__init__(reduction=reduction, name=name)
        self.pcc = PCCLoss(reduction=reduction)


    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        y_pred = math_ops.round(y_pred)
        y_true = math_ops.round(y_true)
        y_pred_mean = math_ops.reduce_mean(y_pred, keepdims=True)
        y_true_mean = math_ops.reduce_mean(y_true, keepdims=True)
        pred_std = math_ops.reduce_std(y_pred, keepdims=True)
        true_std = math_ops.reduce_std(y_true, keepdims=True)

        pearson = self.pcc(y_true, y_pred)
        ccc_n = (2.0 * pearson * pred_std * true_std)

        ccc_d = (math_ops.square(pred_std) + math_ops.square(true_std) + math_ops.square(
            y_pred_mean - y_true_mean))

        ccc = ccc_n / (ccc_d + 1e-25)
        return ccc


class SAGRLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='SAGR'):
        super().__init__(reduction=reduction, name=name)


    def call(self, y_true, y_pred):
        """
            Evaluates the SAGR between estimate and ground truth.
        """
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        sagr = math_ops.cast(tf.math.equal(math_ops.sign(y_pred), math_ops.sign(y_true)), y_pred.dtype)
        sagr = math_ops.reduce_mean(sagr)
        return sagr
