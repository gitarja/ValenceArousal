import tensorflow as tf

class TeacherModel(tf.keras.Model):

    def __init__(self, class_num = 10):
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
    def __init__(self, class_num = 10):
        super(StudentModel, self).__init__(self)
        self.dense1 = tf.keras.layers.Dense(units=64, activation="elu", name="dense1")
        self.dense2 = tf.keras.layers.Dense(units=128, activation="elu", name="dense2")
        self.dense3 = tf.keras.layers.Dense(units=256, activation="elu", name="dense3")
        self.out = tf.keras.layers.Dense(units=class_num, activation=None, name="out")


    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        z = self.out(x)
        return z
