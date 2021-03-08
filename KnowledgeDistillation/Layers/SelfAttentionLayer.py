import tensorflow as tf


#doi: 10.23915/distill.00018
class SelfAttentionLayer2D(tf.keras.layers.Layer):

    def __init__(self, filters=128, **kwargs):
        super(SelfAttentionLayer2D, self).__init__(**kwargs)

        self.f_x = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), activation=None)
        self.g_x = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), activation=None)
        self.h_x = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), activation=None)
        self.v_x = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), activation=None)

        self.softmax = tf.keras.layers.Softmax(axis=-1)
    def call(self, inputs, **kwargs):
        batchsize, width, height, C = inputs.shape
        fx = tf.reshape(self.f_x(inputs), (batchsize, width * height, C))
        gx = tf.reshape(self.g_x(inputs), (batchsize, width * height, C))
        hx = tf.reshape(self.h_x(inputs), (batchsize, width * height, C))

        alpha = self.softmax(tf.matmul(fx, gx, transpose_b=True))
        o = tf.matmul(hx, alpha, transpose_a=True)
        o = self.v_x(tf.reshape(o, (batchsize, width, height, C)))
        return o


#doi: 10.23915/distill.00018
class SelfAttentionLayer1D(tf.keras.layers.Layer):

    def __init__(self, filters=128, **kwargs):
        super(SelfAttentionLayer1D, self).__init__(**kwargs)
        self.filters = filters
        self.f_x = tf.keras.layers.Conv1D(filters, kernel_size=1, activation=None)
        self.g_x = tf.keras.layers.Conv1D(filters, kernel_size=1, activation=None)
        self.h_x = tf.keras.layers.Conv1D(filters, kernel_size=1, activation=None)
        self.v_x = tf.keras.layers.Conv1D(filters, kernel_size=1, activation=None)

        self.softmax = tf.keras.layers.Softmax(axis=-1)
    def call(self, inputs, **kwargs):
        batchsize, n, C = inputs.shape
        fx = self.f_x(inputs)
        gx = self.g_x(inputs)
        hx = self.h_x(inputs)

        alpha = self.softmax(tf.matmul(fx, gx, transpose_b=True))
        o = tf.matmul(hx, alpha, transpose_a=True)
        o = self.v_x(tf.reshape(o, (batchsize, n, self.filters)))
        return o
