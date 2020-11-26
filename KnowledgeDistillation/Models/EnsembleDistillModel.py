import tensorflow as tf
import math



class Baseline(tf.keras.Model):

    def __init__(self, num_output = 4):
        super(Baseline, self).__init__(self)
        self.dense1 = tf.keras.layers.Dense(units=32, activation="elu", name="shared_dense1")
        self.dense2 = tf.keras.layers.Dense(units=64, activation="elu", name="shared_dense2")
        self.dense3 = tf.keras.layers.Dense(units=128, activation="elu", name="shared_dense3")
        self.dense4 = tf.keras.layers.Dense(units=256, activation="elu", name="dense4")
        self.dense5 = tf.keras.layers.Dense(units=512, activation="elu", name="dense5")
        self.logit_val = tf.keras.layers.Dense(units=num_output, activation=None, name="logit_val")
        self.logit_ar = tf.keras.layers.Dense(units=num_output, activation=None, name="logit_ar")

        #dropout
        self.droupout1 = tf.keras.layers.Dropout(0.15, name="dropout1")
        self.droupout2 = tf.keras.layers.Dropout(0.15, name="dropout2")

    def forward(self, x, dense, droput):
        return droput(dense(x))


    def call(self, inputs, training=None, mask=None):
        #shallow
        x = self.forward(inputs, self.dense1, self.droupout1)
        x = self.forward(x, self.dense2, self.droupout1)
        x = self.forward(x, self.dense3, self.droupout1)

        #dense
        x = self.forward(x, self.dense4, self.droupout2)
        x = self.forward(x, self.dense5, self.droupout2)

        z_val = self.logit_val(x)
        z_ar = self.logit_ar(x)

        return z_val, z_ar

class EnsembleTeacher(tf.keras.Model):

    def __init__(self, shallow_outputs=[32, 64, 128], num_output = 4):
        super(EnsembleTeacher, self).__init__(self)
        # shared-shallow layers
        self.shared_dense1 = tf.keras.layers.Dense(units=shallow_outputs[0], activation="elu", name="shared_dense1")
        self.shared_dense2 = tf.keras.layers.Dense(units=shallow_outputs[1], activation="elu", name="shared_dense2")
        self.shared_dense3 = tf.keras.layers.Dense(units=shallow_outputs[2], activation="elu", name="shared_dense3")

        # varied-dense layers
        # 4th layer
        self.dense1_4 = tf.keras.layers.Dense(units=64, activation="elu", name="dense1_4")
        self.dense2_4 = tf.keras.layers.Dense(units=256, activation="elu", name="dense2_4")
        self.dense3_4 = tf.keras.layers.Dense(units=512, activation="elu", name="dense3_4")



        #logit
        self.logit1_val = tf.keras.layers.Dense(units=num_output, activation=None, name="logit1_val")
        self.logit2_val = tf.keras.layers.Dense(units=num_output, activation=None, name="logit2_val")
        self.logit3_val = tf.keras.layers.Dense(units=num_output, activation=None, name="logit3_val")

        self.logit1_ar = tf.keras.layers.Dense(units=num_output, activation=None, name="logit1_ar")
        self.logit2_ar = tf.keras.layers.Dense(units=num_output, activation=None, name="logit2__ar")
        self.logit3_ar = tf.keras.layers.Dense(units=num_output, activation=None, name="logit3_ar")


        #dropout
        self.droupout1 = tf.keras.layers.Dropout(0.05, name="dropout1")
        self.droupout2 = tf.keras.layers.Dropout(0.05, name="dropout2")
        self.droupout3 = tf.keras.layers.Dropout(0.7, name="dropout3")

        #average
        self.average = tf.keras.layers.Average(name="average")


    def forward(self, x, dense, droput):
        return droput(dense(x))

    def call(self, inputs, training=None, mask=None):

        #shared-layer
        x = self.forward(inputs, self.shared_dense1, self.droupout1)
        x = self.forward(x, self.shared_dense2, self.droupout1)
        x = self.forward(x, self.shared_dense3, self.droupout1)

        # model-branching 1
        x1 = self.forward(x, self.dense1_4, self.droupout2)
        logit1_val = self.logit1_val(x1)
        logit1_ar = self.logit1_ar(x1)

        #model-branching 2
        x2 = self.forward(x, self.dense2_4, self.droupout2)
        logit2_val = self.logit2_val(x2)
        logit2_ar = self.logit2_ar(x1)

        #model-branching 3
        x3 = self.forward(x, self.dense3_4, self.droupout2)
        logit3_val = self.logit3_val(x3)
        logit3_ar = self.logit3_ar(x1)

        z_val = self.average([logit1_val, logit2_val, logit3_val])
        z_ar = self.average([logit1_ar, logit2_ar, logit3_ar])

        return z_val, z_ar




class EnsembleStudent(tf.keras.Model):

    def __init__(self, num_output=4):
        super(EnsembleStudent, self).__init__(self)
        self.en_conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=3, activation=None, name="en_conv1",
                                               padding="same")
        self.en_conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=3, activation=None, name="en_conv2",
                                               padding="same")
        self.en_conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=3, activation=None, name="en_conv3",
                                               padding="same")
        self.en_conv4 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=3, activation=None, name="en_conv4",
                                               padding="same")
        self.en_conv5 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=3, activation=None, name="en_conv5",
                                               padding="same")
        self.en_conv6 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=3, activation=None, name="en_conv6",
                                               padding="same")
        self.en_conv7 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=3, activation=None, name="en_conv7",
                                               padding="same")


        #batch normalization
        self.batch_norm1 = tf.keras.layers.BatchNormalization(name="batch_norm1")
        self.batch_norm2 = tf.keras.layers.BatchNormalization(name="batch_norm2")
        self.batch_norm3 = tf.keras.layers.BatchNormalization(name="batch_norm3")
        self.batch_norm4 = tf.keras.layers.BatchNormalization(name="batch_norm4")
        self.batch_norm5 = tf.keras.layers.BatchNormalization(name="batch_norm5")
        self.batch_norm6 = tf.keras.layers.BatchNormalization(name="batch_norm6")
        self.batch_norm7 = tf.keras.layers.BatchNormalization(name="batch_norm7")

        #activation
        self.elu = tf.keras.layers.ELU()

        #logit
        self.logit_ar = tf.keras.layers.Dense(units=num_output, activation=None, name="logit_ar")
        self.logit_val = tf.keras.layers.Dense(units=num_output, activation=None, name="logit_val")

        #flattent
        self.flat = tf.keras.layers.Flatten()



    def forward(self, x, dense, norm, activation):
        return activation(norm(dense(x)))



    def call(self, inputs, training=None, mask=None):
        x = self.forward(inputs, self.en_conv1, self.batch_norm1, self.elu)
        x = self.forward(x, self.en_conv2, self.batch_norm2, self.elu)
        x = self.forward(x, self.en_conv3, self.batch_norm3, self.elu)
        x = self.forward(x, self.en_conv4, self.batch_norm4, self.elu)
        x = self.forward(x, self.en_conv5, self.batch_norm5, self.elu)
        x = self.forward(x, self.en_conv6, self.batch_norm6, self.elu)
        x = self.forward(x, self.en_conv7, self.batch_norm7, self.elu)

        x = self.flat(x)
        z_ar = self.logit_ar(x)
        z_val = self.logit_val(x)

        return z_val, z_ar
