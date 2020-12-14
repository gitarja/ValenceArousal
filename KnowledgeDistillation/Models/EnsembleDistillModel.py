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


        # activation
        self.elu = tf.keras.layers.ELU()

        # classify
        self.class_1 = tf.keras.layers.Dense(units=32, name="class_1")
        self.class_2 = tf.keras.layers.Dense(units=16, name="class_2")

        # logit
        self.logit_ar = tf.keras.layers.Dense(units=num_output, activation=None, name="logit_ar")
        self.logit_val = tf.keras.layers.Dense(units=num_output, activation=None, name="logit_val")

        #dropout
        self.dropout_1 = tf.keras.layers.Dropout(0.3)

        # loss
        self.cross_loss = tf.losses.BinaryCrossentropy(from_logits=True,
                                                       reduction=tf.keras.losses.Reduction.NONE)


    def call(self, inputs, training=None, mask=None):
        z = self.dropout_1(self.elu(self.class_1(inputs)))
        # z = self.dropout_1(self.elu(self.class_2(z)))

        z_ar = self.logit_ar(z)
        z_val = self.logit_val(z)

        return z_ar, z_val


    def train(self, X, y_ar, y_val, th, global_batch_size, training=False):
        z_ar, z_val = self.call(X, training=training)
        final_loss_ar = tf.nn.compute_average_loss(self.cross_loss(y_ar, z_ar), global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(self.cross_loss(y_val, z_val), global_batch_size=global_batch_size)
        predictions_ar = tf.cast(tf.nn.sigmoid(z_ar) >= th, dtype=tf.float32)
        predictions_val = tf.cast(tf.nn.sigmoid(z_val) >= th, dtype=tf.float32)

        final_loss = final_loss_ar + final_loss_val
        return final_loss, predictions_ar, predictions_val


class EnsembleStudentOneDim(tf.keras.Model):

    def __init__(self, num_output=4, pretrain=True):
        super(EnsembleStudentOneDim, self).__init__(self)
        self.en_conv1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="en_conv1",
                                               padding="same", trainable=pretrain)
        self.en_conv2 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="en_conv2",
                                               padding="same", trainable=pretrain)
        self.en_conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="en_conv3",
                                               padding="same", trainable=pretrain)
        self.en_conv4 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="en_conv4",
                                               padding="same", trainable=pretrain)
        self.en_conv5 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="en_conv5",
                                               padding="same", trainable=pretrain)
        self.en_conv6 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="en_conv6",
                                               padding="same", trainable=pretrain)


        #activation
        self.elu = tf.keras.layers.ELU()

        #classify
        self.class_1 = tf.keras.layers.Dense(units=32, name="class_1")


        #logit
        self.logit_ar = tf.keras.layers.Dense(units=num_output, activation=None, name="logit_ar")
        self.logit_val = tf.keras.layers.Dense(units=num_output, activation=None, name="logit_val")


        #flattent
        self.flat = tf.keras.layers.Flatten()

        #pool
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=3)


        #dropout
        self.dropout_1 = tf.keras.layers.Dropout(0.3)

        # loss
        self.cross_loss = tf.losses.BinaryCrossentropy(from_logits=True,
                                                       reduction=tf.keras.losses.Reduction.NONE)
        self.mean_square = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)


    def forward(self, x, dense, norm=None, activation=None):
        if norm is None:
            return activation(dense(x))
        return activation(norm(dense(x)))



    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, -1)

        #encoder
        x = self.max_pool(self.forward(x, self.en_conv1, None, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv2, None, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv3, None, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv4,None, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv5, None, self.elu))
        z = self.max_pool(self.forward(x, self.en_conv6, None, self.elu))


        # print(z.shape)
        z = self.flat(z)
        z = self.dropout_1(self.elu(self.class_1(z)))
        z_ar = self.logit_ar(z)
        z_val = self.logit_val(z)

        return z_ar, z_val


    def train(self, X, y_ar, y_val, th, global_batch_size, training=False):
        z_ar, z_val = self.call(X, training=training)
        final_loss_ar = tf.nn.compute_average_loss(self.cross_loss(y_ar, z_ar), global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(self.cross_loss(y_val, z_val), global_batch_size=global_batch_size)

        predictions_ar = tf.cast(tf.nn.sigmoid(z_ar) >= th, dtype=tf.float32)
        predictions_val = tf.cast(tf.nn.sigmoid(z_val) >= th, dtype=tf.float32)

        final_loss = final_loss_ar + final_loss_val
        return final_loss, predictions_ar, predictions_val

class BaseStudentOneDim(tf.keras.Model):

    def __init__(self, num_output=4, ECG_N= 11125,pretrain=True):
        super(BaseStudentOneDim, self).__init__(self)

        self.en_conv1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="en_conv1",
                                               padding="same", trainable=pretrain, input_shape=(None, ECG_N))
        self.en_conv2 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="en_conv2",
                                               padding="same", trainable=pretrain)
        self.en_conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="en_conv3",
                                               padding="same", trainable=pretrain)
        self.en_conv4 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="en_conv4",
                                               padding="same", trainable=pretrain)
        self.en_conv5 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="en_conv5",
                                               padding="same", trainable=pretrain)
        self.en_conv6 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="en_conv6",
                                               padding="same", trainable=pretrain)

        self.de_conv6 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="de_conv6",
                                               padding="same", trainable=pretrain)
        self.de_conv5 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="de_conv5",
                                               padding="same", trainable=pretrain)
        self.de_conv4 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="de_conv4",
                                               padding="same", trainable=pretrain)
        self.de_conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="de_conv3",
                                               padding="same", trainable=pretrain)
        self.de_conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="de_conv2",
                                               padding="same", trainable=pretrain)
        self.de_conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="de_conv1",
                                               padding="same", trainable=pretrain)
        self.de_conv0 = tf.keras.layers.Conv1D(filters=1, kernel_size=5, strides=1, activation=None, name="de_conv0",
                                               padding="same", trainable=pretrain)




        #activation
        self.elu = tf.keras.layers.ELU()
        self.relu = tf.keras.layers.ReLU()

        #classify
        self.class_1 = tf.keras.layers.Dense(units=64, name="class_1")
        self.class_2 = tf.keras.layers.Dense(units=128, name="class_1")

        #logit
        self.logit = tf.keras.layers.Dense(units=num_output, activation=None, name="logit_ar")


        #flattent
        self.flat = tf.keras.layers.Flatten()

        #pool
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=3)
        self.up_samp = tf.keras.layers.UpSampling1D(size=3)

        #dropout
        self.dropout_1 = tf.keras.layers.Dropout(0.3)


        self.mean_square_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.cross_loss = tf.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)


    def forward(self, x, dense, norm=None, activation=None):
        if norm is None:
            return activation(dense(x))
        return activation(norm(dense(x)))



    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, -1)

        #encoder
        z = self.encode(x)
        x = self.decode(z)


        # print(z.shape)
        # z = self.flat(z)
        # z = self.dropout_1(self.elu(self.class_1(z)))
        # z = self.dropout_1(self.elu(self.class_2(z)))
        # z = self.relu(self.logit(z))


        return x

    def encode(self, x):
        x = self.max_pool(self.forward(x, self.en_conv1, None, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv2, None, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv3, None, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv4, None, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv5, None, self.elu))
        z = self.max_pool(self.forward(x, self.en_conv6, None, self.elu))
        return z

    def decode(self, x):
        x = self.up_samp(self.forward(x, self.de_conv6, None, self.elu))
        x = self.up_samp(self.forward(x, self.de_conv5, None, self.elu))
        x = self.up_samp(self.forward(x, self.de_conv4, None, self.elu))
        x = self.up_samp(self.forward(x, self.de_conv3, None, self.elu))
        x = self.up_samp(self.forward(x, self.de_conv2, None, self.elu))
        x = self.up_samp(self.forward(x, self.de_conv1, None, self.elu))
        z = self.de_conv0(x)
        return z

    # def train(self, X, y,  global_batch_size, training=False):
    #     z = self.call(X, training=training)
    #     final_loss_ar = tf.nn.compute_average_loss(self.mean_square_loss(y, z), global_batch_size=global_batch_size)
    #     return final_loss_ar


    def train(self, X, y,  global_batch_size, training=False):
        z = self.call(X, training=training)
        final_loss_ar = tf.nn.compute_average_loss(self.cross_loss(y, z), global_batch_size=global_batch_size)
        return final_loss_ar

    def extractFeatures(self, inputs):
        x = tf.expand_dims(inputs, -1)
        #encoder
        z = self.encode(x)
        z = self.flat(z)

        return z
    def loadBaseModel(self, checkpoint_prefix):
        model = self
        checkpoint = tf.train.Checkpoint(teacher_model=self)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
        checkpoint.restore(manager.latest_checkpoint)
        model.build(input_shape=(None, 11125))


        return model




