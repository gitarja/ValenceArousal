from tensorflow.keras.metrics import Metric
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np

class PCC(Metric):

    def __init__(self, name="pcc", dtype=None, **kwargs):
        super(PCC, self).__init__(name=name, dtype=dtype, **kwargs)
        self.pcc_r = self.add_weight(name='pcc_r', initializer='zeros')
        self.total_count = self.add_weight(name='total_count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        y_pred_mean = math_ops.reduce_mean(y_pred, keepdims=True)
        y_true_mean = math_ops.reduce_mean(y_true,  keepdims=True)
        y_pred_m = y_pred - y_pred_mean
        y_true_m = y_true - y_true_mean
        y_pred_norm = tf.norm(y_pred_m, keepdims=True)
        y_true_norm = tf.norm(y_true_m, keepdims=True)


        pcc = (math_ops.reduce_sum(y_pred_m * y_true_m, keepdims=True)) / (
                    math_ops.reduce_sum(y_pred_norm * y_true_norm, keepdims=True) + 1e-25)

        pcc = tf.maximum(tf.minimum(pcc, 1.0), -1.0)

        self.pcc_r.assign_add(tf.reduce_sum(pcc))
        self.total_count.assign_add(len(pcc))

    def result(self):
        return math_ops.div_no_nan(self.pcc_r, self.total_count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.pcc_r.assign(0.0)
        self.total_count.assign(0.0)


class CCC(Metric):

    def __init__(self, name="ccc", dtype=None, **kwargs):
        super(CCC, self).__init__(name=name, dtype=dtype, **kwargs)
        self.ccc_r = self.add_weight(name='ccc_r', initializer='zeros')
        self.total_count = self.add_weight(name='total_count', initializer='zeros')
        self.pcc = PCC()

    def update_state(self, y_true, y_pred, sample_weight=None):
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

        ccc = ccc_n / (ccc_d+ 1e-25)
        self.ccc_r.assign_add(tf.reduce_sum(ccc))
        self.total_count.assign_add(len(ccc))


    def result(self):
        return math_ops.div_no_nan(self.ccc_r, self.total_count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.ccc_r.assign(0.0)
        self.total_count.assign(0.0)

class SAGR(Metric):
    def __init__(self, name="sagr", dtype=None, **kwargs):
        super(SAGR, self).__init__(name=name, dtype=dtype, **kwargs)
        self.sagr_r = self.add_weight(name='sagr_r', initializer='zeros')
        self.total_count = self.add_weight(name='total_count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        sagr = math_ops.cast(tf.math.equal(math_ops.sign(y_pred), math_ops.sign(y_true)), y_pred.dtype)
        sagr = math_ops.reduce_mean(sagr, keepdims=True)
        self.sagr_r.assign_add(tf.reduce_sum(sagr))
        self.total_count.assign_add(len(sagr))

    def result(self):
        return math_ops.div_no_nan(self.sagr_r, self.total_count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.sagr_r.assign(0.0)
        self.total_count.assign(0.0)

class SoftF1(Metric):
    def __init__(self, name="softf1", dtype=None, **kwargs):
        super(SoftF1, self).__init__(name=name, dtype=dtype, **kwargs)
        self.softf1_r = self.add_weight(name='softf1_r', initializer='zeros')
        self.total_count = self.add_weight(name='total_count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp = math_ops.reduce_sum(y_pred * y_true, axis=-1, keepdims=True)  # compute soft tp
        fp = math_ops.reduce_sum(y_pred * (1 - y_true), axis=-1, keepdims=True)  # compute soft fp
        fn = math_ops.reduce_sum((1 - y_pred) * y_true, axis=-1, keepdims=True)  # compute soft fn
        tn = math_ops.reduce_sum((1 - y_pred) * (1 - y_true), axis=-1, keepdims=True)  # compute soft tn

        s_p = 2 * tp / (2 * tp + fn + fp + 1e-16)
        s_n = 2 * tn / (2 * tn + fn + fp + 1e-16)

        loss_p = s_p
        loss_n = s_n

        cost = 0.5 * (loss_p + loss_n)

        self.softf1_r.assign_add(tf.reduce_sum(cost))
        self.total_count.assign_add(len(cost))

    def result(self):
        return math_ops.div_no_nan(self.softf1_r, self.total_count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.softf1_r.assign(0.0)
        self.total_count.assign(0.0)