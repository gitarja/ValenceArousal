import tensorflow as tf
import math
from KnowledgeDistillation.Layers.AttentionLayer import AttentionLayer

class EnsembleStudentOneDimF(tf.keras.Model):

    def __init__(self, num_output_ar=4, num_output_val=4, pretrain=True):
        super(EnsembleStudentOneDimF, self).__init__(self)

        # dense 1
        self.small_en_1 = tf.keras.layers.Dense(units=16, name="small_en_1")
        self.small_en_2 = tf.keras.layers.Dense(units=32, name="small_en_2")
        self.small_en_3 = tf.keras.layers.Dense(units=64, name="small_en_3")
        self.small_ar_logit = tf.keras.layers.Dense(units=num_output_ar, name="small_ar_logit", activation=None)
        self.small_val_logit = tf.keras.layers.Dense(units=num_output_val, name="small_val_logit", activation=None)

        # dense 2
        self.med_en_1 = tf.keras.layers.Dense(units=16, name="med_en_1")
        self.med_en_2 = tf.keras.layers.Dense(units=16, name="med_en_2")
        self.med_en_3 = tf.keras.layers.Dense(units=32, name="med_en_3")
        self.med_en_4 = tf.keras.layers.Dense(units=32, name="med_en_4")
        self.med_ar_logit = tf.keras.layers.Dense(units=num_output_ar, name="med_ar_logit", activation=None)
        self.med_val_logit = tf.keras.layers.Dense(units=num_output_val, name="med_val_logit", activation=None)

        # big
        self.large_en_1 = tf.keras.layers.Dense(units=32, name="large_en_1")
        self.large_en_2 = tf.keras.layers.Dense(units=64, name="large_en_2")
        self.large_en_3 = tf.keras.layers.Dense(units=64, name="large_en_3")
        self.large_ar_logit = tf.keras.layers.Dense(units=num_output_ar, name="large_ar_logit", activation=None)
        self.large_val_logit = tf.keras.layers.Dense(units=num_output_val, name="large_val_logit", activation=None)


       # activation
        self.elu = tf.keras.layers.ELU()
        # dropout
        self.dropout1 = tf.keras.layers.Dropout(0.0)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        # avg
        self.avg = tf.keras.layers.Average()

        # loss
        self.cross_loss = tf.losses.BinaryCrossentropy(from_logits=True,
                                                                  reduction=tf.keras.losses.Reduction.NONE)


    def smallForward(self, x):
        x = self.dropout2(self.elu(self.small_en_1(x)))
        x = self.dropout2(self.elu(self.small_en_2(x)))
        x = self.dropout2(self.elu(self.small_en_3(x)))
        ar_logit = self.small_ar_logit(x)
        val_logit = self.small_val_logit(x)
        return ar_logit, val_logit

    def mediumForward(self, x):
        x = self.dropout2(self.elu(self.med_en_1(x)))
        x = self.dropout2(self.elu(self.med_en_2(x)))
        x = self.dropout2(self.elu(self.med_en_3(x)))
        x = self.dropout2(self.elu(self.med_en_4(x)))
        ar_logit = self.med_ar_logit(x)
        val_logit = self.med_val_logit(x)

        return ar_logit, val_logit

    def largeForward(self, x):
        x = self.dropout2(self.elu(self.large_en_1(x)))
        x = self.dropout2(self.elu(self.large_en_2(x)))
        x = self.dropout2(self.elu(self.large_en_3(x)))
        ar_logit = self.large_ar_logit(x)
        val_logit = self.large_val_logit(x)

        return ar_logit, val_logit
    def call(self, inputs, training=None, mask=None):
        # small
        ar_logit_small, val_logit_small = self.smallForward(inputs)
        # med
        ar_logit_med, val_logit_med = self.mediumForward(inputs)
        # big
        ar_logit_large, val_logit_large = self.largeForward(inputs)
        ar_logits = self.avg([ar_logit_small, ar_logit_med, ar_logit_large])
        val_logits = self.avg([val_logit_small, val_logit_med, val_logit_large])

        return ar_logits, val_logits

    @tf.function
    def trainM(self, X, y_ar, y_val, y_ar_t, y_val_t, th, c_f, alpha, global_batch_size, training=False):
        z_ar, z_val  = self.call(X, training=training)
        y_ar_t = tf.nn.sigmoid(y_ar_t)
        y_val_t = tf.nn.sigmoid(y_val_t)
        beta = 1 - alpha
        final_loss_ar = tf.nn.compute_average_loss(
            (alpha * self.cross_loss(y_ar, z_ar)) + (beta * self.cross_loss(y_ar_t, z_ar)), sample_weight=c_f,
            global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(
            (alpha * self.cross_loss(y_val, z_val)) + (beta * self.cross_loss(y_val_t, z_val)), sample_weight=c_f,
            global_batch_size=global_batch_size)
        predictions_ar = tf.cast(tf.nn.sigmoid(z_ar) >= th, dtype=tf.float32)
        predictions_val = tf.cast(tf.nn.sigmoid(z_val) >= th, dtype=tf.float32)

        # final_loss = (final_loss_ar + final_loss_val)
        return final_loss_ar, final_loss_val, predictions_ar, predictions_val

    @tf.function
    def test(self, X, y_ar, y_val, th, c_f, global_batch_size, training=False):
        z_ar, z_val = self.call(X, training=training)

        final_loss_ar = tf.nn.compute_average_loss(self.cross_loss(y_ar, z_ar), sample_weight=c_f, global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(self.cross_loss(y_val, z_val),  sample_weight=c_f, global_batch_size=global_batch_size)

        predictions_ar = tf.cast(tf.nn.sigmoid(z_ar) >= th, dtype=tf.float32)
        predictions_val = tf.cast(tf.nn.sigmoid(z_val) >= th, dtype=tf.float32)

        # final_loss = final_loss_ar + final_loss_val
        return final_loss_ar, final_loss_val, predictions_ar, predictions_val

    @tf.function
    def predict(self, X, global_batch_size, training=False):
        z_ar, z_val = self.call(X, training=training)
        predictions_ar = tf.nn.sigmoid(z_ar)
        predictions_val = tf.nn.sigmoid(z_val)

        return predictions_ar, predictions_val


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

        #attention
        self.att_ar = AttentionLayer(name="att_ar", TIME_STEPS=12)
        self.att_val = AttentionLayer(name="att_val", TIME_STEPS=12)

        # classify
        self.class_ar = tf.keras.layers.Dense(units=32, name="class_ar")
        self.class_val = tf.keras.layers.Dense(units=32, name="class_val")


        self.class_ar_h2 = tf.keras.layers.Dense(units=64, name="class_ar_h2")
        self.class_val_h2 = tf.keras.layers.Dense(units=64, name="class_val_h2")

        self.class_ar_h3 = tf.keras.layers.Dense(units=128, name="class_ar_h3")
        self.class_val_h3 = tf.keras.layers.Dense(units=128, name="class_val_h3")

        # logit
        self.logit_ar = tf.keras.layers.Dense(units=num_output_ar, activation=None, name="logit_ar")
        self.logit_val = tf.keras.layers.Dense(units=num_output_val, activation=None, name="logit_val")

        self.logit_ar_h2 = tf.keras.layers.Dense(units=num_output_ar, activation=None, name="logit_ar_h2")
        self.logit_val_h2 = tf.keras.layers.Dense(units=num_output_val, activation=None, name="logit_val_h2")

        self.logit_ar_h3 = tf.keras.layers.Dense(units=num_output_ar, activation=None, name="logit_ar_h3")
        self.logit_val_h3 = tf.keras.layers.Dense(units=num_output_val, activation=None, name="logit_val_h3")

        # flattent
        self.flat = tf.keras.layers.Flatten()

        # pool
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=3)

        # dropout
        self.dropout_1 = tf.keras.layers.Dropout(0.3)

        # avg
        self.avg = tf.keras.layers.Average()
        # loss
        self.cross_loss = tf.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.2,
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

        # flat logit
        z_ar = self.att_ar(z)
        z_val = self.att_val(z)

        z_ar = self.dropout_1(self.flat(z_ar))
        z_val = self.dropout_1(self.flat(z_val))

        #head 1
        z_ar_h1 = self.elu(self.class_ar(z_ar))
        z_val_h1 = self.elu(self.class_val(z_val))

        z_ar_h1 = self.logit_ar(z_ar_h1)
        z_val_h1 = self.logit_val(z_val_h1)

        #head 2

        z_ar_h2 = self.elu(self.class_ar_h2(z_ar))
        z_val_h2 = self.elu(self.class_val_h2(z_val))

        z_ar_h2 = self.logit_ar_h2(z_ar_h2)
        z_val_h2 = self.logit_val_h2(z_val_h2)

        # head 3

        z_ar_h3 = self.elu(self.class_ar_h3(z_ar))
        z_val_h3 = self.elu(self.class_val_h3(z_val))

        z_ar_h3 = self.logit_ar_h3(z_ar_h3)
        z_val_h3 = self.logit_val_h3(z_val_h3)

        z_ar = self.avg([z_ar_h1, z_ar_h2, z_ar_h3])
        z_val = self.avg([z_val_h1, z_val_h2, z_val_h3])

        return z_ar, z_val, z

    @tf.function
    def trainM(self, X, y_ar, y_val, y_ar_t, y_val_t, th, c_f, alpha, global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)
        y_ar_t = tf.nn.sigmoid(y_ar_t)
        y_val_t = tf.nn.sigmoid(y_val_t)
        beta = 1 - alpha
        final_loss_ar = tf.nn.compute_average_loss(
            (alpha * self.cross_loss(y_ar, z_ar)) + (beta * self.cross_loss(y_ar_t, z_ar)), sample_weight=c_f,
            global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(
            (alpha * self.cross_loss(y_val, z_val)) + (beta * self.cross_loss(y_val_t, z_val)), sample_weight=c_f,
            global_batch_size=global_batch_size)
        predictions_ar = tf.cast(tf.nn.sigmoid(z_ar) >= th, dtype=tf.float32)
        predictions_val = tf.cast(tf.nn.sigmoid(z_val) >= th, dtype=tf.float32)

        # final_loss = (final_loss_ar + final_loss_val)
        return final_loss_ar, final_loss_val, predictions_ar, predictions_val

    @tf.function
    def test(self, X, y_ar, y_val, th, c_f, global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)

        final_loss_ar = tf.nn.compute_average_loss(self.cross_loss(y_ar, z_ar), sample_weight=c_f, global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(self.cross_loss(y_val, z_val),  sample_weight=c_f, global_batch_size=global_batch_size)

        predictions_ar = tf.cast(tf.nn.sigmoid(z_ar) >= th, dtype=tf.float32)
        predictions_val = tf.cast(tf.nn.sigmoid(z_val) >= th, dtype=tf.float32)

        # final_loss = final_loss_ar + final_loss_val
        return final_loss_ar, final_loss_val, predictions_ar, predictions_val

    @tf.function
    def predict(self, X, global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)
        predictions_ar = tf.nn.sigmoid(z_ar)
        predictions_val = tf.nn.sigmoid(z_val)

        return predictions_ar, predictions_val


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


        #attention
        self.att_ar = AttentionLayer(name="att_ar", TIME_STEPS=15)
        self.att_val = AttentionLayer(name="att_val", TIME_STEPS=15)

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

        self.multi_cross_loss = tf.losses.BinaryCrossentropy(from_logits=True,
                                                             reduction=tf.keras.losses.Reduction.NONE)

        self.kld_loss = tf.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)

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


        #flat logit
        z_ar = self.att_ar(z)
        z_val = self.att_val(z)

        z_ar = self.dropout_1(self.flat(z_ar))
        z_val = self.dropout_1(self.flat(z_val))

        z_ar = self.elu(self.class_ar(z_ar))
        z_val = self.elu(self.class_val(z_val))


        z_ar = self.logit_ar(z_ar)
        z_val = self.logit_val(z_val)

        return z_ar, z_val, z

    @tf.function
    def trainM(self, X, y_ar, y_val, y_ar_t, y_val_t, T, alpha, global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)
        y_ar_t = tf.nn.softmax(y_ar_t / T, -1)
        y_val_t = tf.nn.softmax(y_val_t / T, -1)
        beta = 1 - alpha
        final_loss_ar = tf.nn.compute_average_loss((alpha * self.multi_cross_loss(y_ar, z_ar)) + (
                    beta * self.multi_cross_loss(y_ar_t, z_ar)), global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(
            (alpha * self.multi_cross_loss(y_val, z_val)) + (
                    beta * self.multi_cross_loss(y_val_t, z_val)), global_batch_size=global_batch_size)

        prediction_ar = tf.nn.sigmoid(z_ar)
        prediction_val = tf.nn.sigmoid(z_val)

        final_loss = (final_loss_ar + final_loss_val)
        return final_loss, prediction_ar, prediction_val

    @tf.function
    def test(self, X, y_ar, y_val, global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)

        final_loss_ar = tf.nn.compute_average_loss(self.multi_cross_loss(y_ar,  z_ar),
                                                   global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(self.multi_cross_loss(y_val,  z_val),
                                                    global_batch_size=global_batch_size)
        prediction_ar = tf.nn.sigmoid(z_ar)
        prediction_val = tf.nn.sigmoid(z_val)
        final_loss = final_loss_ar + final_loss_val
        return final_loss, prediction_ar, prediction_val
