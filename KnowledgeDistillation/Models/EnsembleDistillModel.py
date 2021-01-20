import tensorflow as tf
import math
from KnowledgeDistillation.Layers.AttentionLayer import AttentionLayer


class EnsembleStudentOneDim(tf.keras.Model):

    def __init__(self, num_output_ar=4, num_output_val=4, pretrain=True):
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

        self.batch_1 = tf.keras.layers.BatchNormalization(name="batch_1")
        self.batch_2 = tf.keras.layers.BatchNormalization(name="batch_2")
        self.batch_3 = tf.keras.layers.BatchNormalization(name="batch_3")
        self.batch_4 = tf.keras.layers.BatchNormalization(name="batch_4")
        self.batch_5 = tf.keras.layers.BatchNormalization(name="batch_5")
        self.batch_6 = tf.keras.layers.BatchNormalization(name="batch_6")

        # activation
        self.elu = tf.keras.layers.ELU()

        # classify
        self.class_ar = tf.keras.layers.Dense(units=32, name="class_ar")
        self.class_val = tf.keras.layers.Dense(units=32, name="class_val")

        # logit
        self.logit_ar = tf.keras.layers.Dense(units=num_output_ar, activation=None, name="logit_ar")
        self.logit_val = tf.keras.layers.Dense(units=num_output_val, activation=None, name="logit_val")

        # flattent
        self.flat = tf.keras.layers.Flatten()

        # pool
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=3)

        # dropout
        self.dropout_1 = tf.keras.layers.Dropout(0.3)

        # loss
        # loss
        self.cross_loss = tf.losses.BinaryCrossentropy(from_logits=True,
                                                       reduction=tf.keras.losses.Reduction.NONE)
        self.multi_cross_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                        reduction=tf.keras.losses.Reduction.NONE)
        self.mean_square_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.cos_loss = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)

    def forward(self, x, dense, norm=None, activation=None):
        if norm is None:
            return activation(dense(x))
        return activation(norm(dense(x)))

    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, -1)

        # encoder
        x = self.max_pool(self.forward(x, self.en_conv1, self.batch_1, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv2, self.batch_2, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv3, self.batch_3, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv4, self.batch_4, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv5, self.batch_5, self.elu))
        z = self.max_pool(self.forward(x, self.en_conv6, self.batch_6, self.elu))

        # print(z.shape)
        logit = self.dropout_1(self.flat(z))
        z_ar = self.elu(self.class_ar(logit))
        z_val = self.elu(self.class_val(logit))

        z_ar = self.logit_ar(z_ar)
        z_val = self.logit_val(z_val)

        return z_ar, z_val, z

    @tf.function
    def trainM(self, X, y_ar, y_val, y_ar_t, y_val_t, th, alpha, global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)
        y_ar_t = tf.nn.sigmoid(y_ar_t)
        y_val_t = tf.nn.sigmoid(y_val_t)
        beta = 1 - alpha
        final_loss_ar = tf.nn.compute_average_loss(
            (alpha * self.cross_loss(y_ar, z_ar)) + (beta * self.cross_loss(y_ar_t, z_ar)),
            global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(
            (alpha * self.cross_loss(y_val, z_val)) + (beta * self.cross_loss(y_val_t, z_val)),
            global_batch_size=global_batch_size)
        predictions_ar = tf.cast(tf.nn.sigmoid(z_ar) >= th, dtype=tf.float32)
        predictions_val = tf.cast(tf.nn.sigmoid(z_val) >= th, dtype=tf.float32)

        final_loss = (final_loss_ar + final_loss_val)
        return final_loss, predictions_ar, predictions_val

    @tf.function
    def test(self, X, y_ar, y_val, th, global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)

        final_loss_ar = tf.nn.compute_average_loss(self.cross_loss(y_ar, z_ar), global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(self.cross_loss(y_val, z_val), global_batch_size=global_batch_size)

        predictions_ar = tf.cast(tf.nn.sigmoid(z_ar) >= th, dtype=tf.float32)
        predictions_val = tf.cast(tf.nn.sigmoid(z_val) >= th, dtype=tf.float32)

        final_loss = final_loss_ar + final_loss_val
        return final_loss, predictions_ar, predictions_val


class EnsembleStudentOneDim_MClass(tf.keras.Model):

    def __init__(self, num_output_val=3, num_output_ar=2, pretrain=True):
        super(EnsembleStudentOneDim_MClass, self).__init__(self)
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

        self.batch_1 = tf.keras.layers.BatchNormalization(name="batch_1")
        self.batch_2 = tf.keras.layers.BatchNormalization(name="batch_2")
        self.batch_3 = tf.keras.layers.BatchNormalization(name="batch_3")
        self.batch_4 = tf.keras.layers.BatchNormalization(name="batch_4")
        self.batch_5 = tf.keras.layers.BatchNormalization(name="batch_5")
        self.batch_6 = tf.keras.layers.BatchNormalization(name="batch_6")

        # activation
        self.elu = tf.keras.layers.ELU()

        # dense 1
        self.class_ar = tf.keras.layers.Dense(units=32, name="class_ar")
        self.class_val = tf.keras.layers.Dense(units=32, name="class_val")


        # logit

        self.logit_ar = tf.keras.layers.Dense(units=num_output_ar, activation=None, name="logit_ar")
        self.logit_val = tf.keras.layers.Dense(units=num_output_val, activation=None, name="logit_val")

        # flattent
        self.flat = tf.keras.layers.Flatten()

        # pool
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=3)

        # dropout
        self.dropout_1 = tf.keras.layers.Dropout(0.15)


        # loss

        self.multi_cross_loss = tf.losses.BinaryCrossentropy(from_logits=False,
                                                             reduction=tf.keras.losses.Reduction.NONE)
        self.mean_square_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def forward(self, x, dense, norm=None, activation=None):
        if norm is None:
            return activation(dense(x))
        return activation(norm(dense(x)))

    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, -1)

        # encoder ar
        x = self.max_pool(self.forward(x, self.en_conv1, self.batch_1, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv2, self.batch_2, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv3, self.batch_3, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv4, self.batch_4, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv5, self.batch_5, self.elu))
        z = self.max_pool(self.forward(x, self.en_conv6, self.batch_6, self.elu))



        # print(z.shape)

        #flat logit
        z_ar = self.dropout_1(self.flat(z))
        z_val = self.dropout_1(self.flat(z))

        z_ar = self.elu(self.class_ar(z_ar))
        z_val = self.elu(self.class_val(z_val))


        z_ar = self.logit_ar(z_ar)
        z_val = self.logit_val(z_val)

        return z_ar, z_val, z

    @tf.function
    def trainM(self, X, y_ar, y_val, y_ar_t, y_val_t, T, alpha, curriculum_weight, global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)
        y_ar_t = tf.nn.softmax(y_ar_t / T, -1)
        y_val_t = tf.nn.softmax(y_val_t / T, -1)
        beta = 1 - alpha
        final_loss_ar = tf.nn.compute_average_loss((alpha * self.multi_cross_loss(y_ar, tf.nn.sigmoid(z_ar))) + (
                    beta * self.multi_cross_loss(y_ar_t, tf.nn.sigmoid(z_ar))), sample_weight=curriculum_weight, global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(
            (alpha * self.multi_cross_loss(y_val, tf.nn.sigmoid(z_val))) + (
                    beta * self.multi_cross_loss(y_val_t, tf.nn.sigmoid(z_val))), sample_weight=curriculum_weight, global_batch_size=global_batch_size)

        prediction_ar = tf.nn.sigmoid(z_ar)
        prediction_val = tf.nn.sigmoid(z_val)

        final_loss = (final_loss_ar + final_loss_val)
        return final_loss, prediction_ar, prediction_val

    @tf.function
    def test(self, X, y_ar, y_val, curriculum_weight, global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)

        final_loss_ar = tf.nn.compute_average_loss(self.multi_cross_loss(y_ar,  tf.nn.sigmoid(z_ar)), sample_weight=curriculum_weight,
                                                   global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(self.multi_cross_loss(y_val,  tf.nn.sigmoid(z_val)), sample_weight=curriculum_weight,
                                                    global_batch_size=global_batch_size)
        prediction_ar = tf.nn.sigmoid(z_ar)
        prediction_val = tf.nn.sigmoid(z_val)
        final_loss = final_loss_ar + final_loss_val
        return final_loss, prediction_ar, prediction_val
