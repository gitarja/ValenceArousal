import tensorflow as tf
import math


class TeacherModel(tf.keras.Model):

    def __init__(self, class_num=10):
        super(TeacherModel, self).__init__(self)
        self.dense1 = tf.keras.layers.Dense(units=128, activation="elu", name="dense1")
        self.dense2 = tf.keras.layers.Dense(units=256, activation="elu", name="dense2")
        self.dense3 = tf.keras.layers.Dense(units=512, activation="elu", name="dense3")
        self.out = tf.keras.layers.Dense(units=class_num, activation=None, name="out")

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        z = self.out(x)

        return z


class StudentModel(tf.keras.Model):
    def __init__(self, class_num=10):
        super(StudentModel, self).__init__(self)

        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, name="conv1", activation="relu")
        self.conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, name="conv2", activation="relu")
        self.conv3 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, name="conv3", activation="relu")
        self.logit = tf.keras.layers.Dense(units=class_num)

        self.drop_out = tf.keras.layers.Dropout(0.5, training=True)
        self.time_series = tf.keras.layers.TimeDistributed()

    def call(self, inputs, training=None, mask=None):
        x = self.drop_out(self.conv1(inputs))
        x = self.drop_out(self.conv2(x))
        x = self.drop_out(self.conv3(x))
        z = self.time_series(self.logit(x))

        return z

    def computeLostt(self, X, y, mean_ytrain, std_ytrain, reg_val, T):
        z = self.call(X)

        rmse_standard_pred = tf.reduce_mean((y - z) ** 2.) ** 0.5

        y_hat = (z * std_ytrain) + mean_ytrain
        mc_pred = tf.reduce_mean(y_hat, axis=0)

        rmse = tf.reduce_mean((y - mc_pred) ** 2.) ** 0.5
        l1 = tf.reduce_logsumexp((-0.5 * reg_val * (y - y_hat) ** 2., 0) - tf.math.log(T) - 0.5 * tf.math.log(
            2 * math.pi) + 0.5 * tf.math.log(reg_val))

        return rmse_standard_pred, rmse, l1
