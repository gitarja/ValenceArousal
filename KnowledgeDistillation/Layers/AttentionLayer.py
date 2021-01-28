import tensorflow as tf


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, TIME_STEPS=15, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.W1 = tf.keras.layers.Dense(TIME_STEPS,  activation='softmax')
        self.permute = tf.keras.layers.Permute((2, 1))
        self.multi = tf.keras.layers.Multiply()

    def call(self, inputs, **kwargs):
        # inputs.shape = (batch_size, time_steps, input_dim)
        a = self.permute(inputs)
        a = self.W1(a)
        a_probs = self.permute(a)
        output_attention_mul = self.multi([inputs, a_probs])

        return output_attention_mul