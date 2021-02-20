import tensorflow as tf
import math
from KnowledgeDistillation.Layers.AttentionLayer import AttentionLayer
from KnowledgeDistillation.Utils.Losses import PCCLoss, CCCLoss


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
        z_ar, z_val = self.call(X, training=training)
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

        final_loss_ar = tf.nn.compute_average_loss(self.cross_loss(y_ar, z_ar), sample_weight=c_f,
                                                   global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(self.cross_loss(y_val, z_val), sample_weight=c_f,
                                                    global_batch_size=global_batch_size)

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
        self.en_conv1 = tf.keras.layers.Conv1D(filters=4, kernel_size=5, strides=1, activation=None, name="en_conv1",
                                               padding="same", trainable=pretrain)
        self.en_conv2 = tf.keras.layers.Conv1D(filters=4, kernel_size=5, strides=1, activation=None, name="en_conv2",
                                               padding="same", trainable=pretrain)
        self.en_conv3 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="en_conv3",
                                               padding="same", trainable=pretrain)
        self.en_conv4 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="en_conv4",
                                               padding="same", trainable=pretrain)
        self.en_conv5 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="en_conv5",
                                               padding="same", trainable=pretrain)
        self.en_conv6 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="en_conv6",
                                               padding="same", trainable=pretrain)

        self.batch_1 = tf.keras.layers.BatchNormalization(name="batch_1")
        self.batch_2 = tf.keras.layers.BatchNormalization(name="batch_2")
        self.batch_3 = tf.keras.layers.BatchNormalization(name="batch_3")
        self.batch_4 = tf.keras.layers.BatchNormalization(name="batch_4")
        self.batch_5 = tf.keras.layers.BatchNormalization(name="batch_5")
        self.batch_6 = tf.keras.layers.BatchNormalization(name="batch_6")

        # activation
        self.elu = tf.keras.layers.ELU()

        # attention
        self.att_ar = AttentionLayer(name="att_ar", TIME_STEPS=15)
        self.att_val = AttentionLayer(name="att_val", TIME_STEPS=15)

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

        # logit regression
        # logit
        self.logit_ar_r = tf.keras.layers.Dense(units=num_output_ar, activation=None, name="logit_ar_r")
        self.logit_val_r = tf.keras.layers.Dense(units=num_output_val, activation=None, name="logit_val_r")

        self.logit_ar_h2_r = tf.keras.layers.Dense(units=num_output_ar, activation=None, name="logit_ar_h2_r")
        self.logit_val_h2_r = tf.keras.layers.Dense(units=num_output_val, activation=None, name="logit_val_h2_r")

        self.logit_ar_h3_r = tf.keras.layers.Dense(units=num_output_ar, activation=None, name="logit_ar_h3_r")
        self.logit_val_h3_r = tf.keras.layers.Dense(units=num_output_val, activation=None, name="logit_val_h3_r")

        # flattent
        self.flat = tf.keras.layers.Flatten()

        # pool
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=3)
        self.avg_pool = tf.keras.layers.AveragePooling1D(pool_size=3)

        # dropout
        self.dropout_1 = tf.keras.layers.Dropout(0.2)
        self.spatial_dropout = tf.keras.layers.SpatialDropout1D(0.2)

        # avg
        self.avg = tf.keras.layers.Average()
        # loss
        self.cross_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                       reduction=tf.keras.losses.Reduction.NONE)
        self.mse_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.pcc_loss = PCCLoss(reduction=tf.keras.losses.Reduction.NONE)
        self.ccc_loss = CCCLoss(reduction=tf.keras.losses.Reduction.NONE)

    def forward(self, x, dense, norm=None, activation=None, training=False):
        if norm is None:
            return activation(dense(x))
        return activation(norm(dense(x), training=training))

    def call(self, inputs, training=None, mask=None):
        # z = inputs
        # x = tf.expand_dims(inputs, -1)

        # encoder
        x = self.max_pool(self.forward(inputs, self.en_conv1, self.batch_1, self.elu, training))
        # x = self.spatial_dropout(x, training=training)
        x = self.max_pool(self.forward(x, self.en_conv2, self.batch_2, self.elu, training))
        # x = self.spatial_dropout(x, training=training)
        x = self.max_pool(self.forward(x, self.en_conv3, self.batch_3, self.elu, training))
        x = self.max_pool(self.forward(x, self.en_conv4, self.batch_4, self.elu, training))
        x = self.max_pool(self.forward(x, self.en_conv5, self.batch_5, self.elu, training))
        z = self.max_pool(self.forward(x, self.en_conv6, self.batch_6, self.elu, training))

        # flat logit
        z_ar = self.spatial_dropout(self.att_ar(z), training=training)
        z_val = self.spatial_dropout(self.att_val(z), training=training)

        z_ar = self.flat(z_ar)
        z_val = self.flat(z_val)

        # head 1
        z_ar_h1 = self.elu(self.class_ar(z_ar))
        z_val_h1 = self.elu(self.class_val(z_val))

        # reg
        z_ar_h1_r = self.logit_ar_r(z_ar_h1)
        z_val_h1_r = self.logit_val_r(z_val_h1)

        z_ar_h1 = self.logit_ar(z_ar_h1)
        z_val_h1 = self.logit_val(z_val_h1)

        # head 2

        z_ar_h2 = self.elu(self.class_ar_h2(z_ar))
        z_val_h2 = self.elu(self.class_val_h2(z_val))

        # reg
        z_ar_h2_r = self.logit_ar_h2_r(z_ar_h2)
        z_val_h2_r = self.logit_ar_h2_r(z_val_h2)

        z_ar_h2 = self.logit_ar_h2(z_ar_h2)
        z_val_h2 = self.logit_val_h2(z_val_h2)

        # head 3

        z_ar_h3 = self.elu(self.class_ar_h3(z_ar))
        z_val_h3 = self.elu(self.class_val_h3(z_val))

        # reg
        z_ar_h3_r = self.logit_ar_h3_r(z_ar_h3)
        z_val_h3_r = self.logit_ar_h3_r(z_val_h3)

        z_ar_h3 = self.logit_ar_h3(z_ar_h3)
        z_val_h3 = self.logit_val_h3(z_val_h3)

        z_ar = self.avg([z_ar_h1, z_ar_h2, z_ar_h3])
        z_val = self.avg([z_val_h1, z_val_h2, z_val_h3])
        # regression
        z_ar_r = self.avg([z_ar_h1_r, z_ar_h2_r, z_ar_h3_r])
        z_val_r = self.avg([z_val_h1_r, z_val_h2_r, z_val_h3_r])

        return z_ar, z_val, z_ar_r, z_val_r

    @tf.function
    def trainM(self, X, y_d_ar, y_d_val, y_ar_t, y_val_t, y_r_ar, y_r_val, th, ar_weight, val_weight, alpha,
               global_batch_size, training=True):
        z_ar, z_val, z_ar_r, z_val_r = self.call(X, training=training)
        y_ar_t = tf.nn.sigmoid(y_ar_t)
        y_val_t = tf.nn.sigmoid(y_val_t)
        beta = 1 - alpha
        final_loss_ar = tf.nn.compute_average_loss(
            (alpha * self.cross_loss(y_d_ar, z_ar)) + (beta * self.cross_loss(y_ar_t, z_ar)),
            global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(
            (alpha * self.cross_loss(y_d_val, z_val)) + (beta * self.cross_loss(y_val_t, z_val)),
            global_batch_size=global_batch_size)

        # regression loss
        mse_loss = tf.nn.compute_average_loss(0.5 * (self.mse_loss(y_r_ar, z_ar_r) + self.mse_loss(y_r_val, z_val_r)),
                                              global_batch_size=global_batch_size)
        pcc_loss = tf.nn.compute_average_loss(
            1 - (0.5 * (self.pcc_loss(y_r_ar, z_ar_r) + self.pcc_loss(y_r_val, z_val_r))),
            global_batch_size=global_batch_size)

        reg_loss = 0.5 * (mse_loss + pcc_loss)

        return final_loss_ar, final_loss_val, reg_loss, z_ar_r, z_val_r

    @tf.function
    def classificationLoss(self, z_ar, z_val, y_d_ar, y_d_val, y_ar_t, y_val_t, alpha, global_batch_size):
        y_ar_t = tf.nn.sigmoid(y_ar_t)
        y_val_t = tf.nn.sigmoid(y_val_t)
        beta = 1 - alpha
        final_loss_ar = tf.nn.compute_average_loss(
            (alpha * self.cross_loss(y_d_ar, z_ar)) + (beta * self.cross_loss(y_ar_t, z_ar)),
            global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(
            (alpha * self.cross_loss(y_d_val, z_val)) + (beta * self.cross_loss(y_val_t, z_val)),
            global_batch_size=global_batch_size)

        return (final_loss_ar + final_loss_val)

    @tf.function
    def regressionLoss(self,  z_r_ar, z_r_val, y_r_ar, y_r_val, shake_params, global_batch_size):
        a = shake_params[0] / tf.reduce_sum(shake_params)
        b = shake_params[1] / tf.reduce_sum(shake_params)
        t = shake_params[2] / tf.reduce_sum(shake_params)

        mse_loss = tf.nn.compute_average_loss(self.mse_loss(y_r_ar, z_r_ar) + self.mse_loss(y_r_val, z_r_val),
                                              global_batch_size=global_batch_size)
        pcc_loss = tf.nn.compute_average_loss(
            1 - (0.5 * (self.pcc_loss(y_r_ar, z_r_ar) + self.pcc_loss(y_r_val, z_r_val))),
            global_batch_size=global_batch_size)
        ccc_loss = tf.nn.compute_average_loss(
            1 - (0.5 * (self.ccc_loss(y_r_ar, z_r_ar) + self.ccc_loss(y_r_val, z_r_val))),
            global_batch_size=global_batch_size)

        return (a * mse_loss) + (b*pcc_loss) + (t*ccc_loss)


    @tf.function
    def predict(self, X, global_batch_size, training=False):
        z_ar, z_val, z_ar_r, z_val_r = self.call(X, training=training)
        predictions_ar = tf.nn.sigmoid(z_ar)
        predictions_val = tf.nn.sigmoid(z_val)

        return predictions_ar, predictions_val


    def symmtericLoss(self, t, y, alpha=6.0, beta=1.):
        t2 = tf.clip_by_value(t, 1e-4, 1.0)
        y2 = tf.clip_by_value(y, 1e-7, 1.0)
        loss = alpha * self.cross_loss(t, y) + beta * self.cross_loss(y2, t2)
        return loss
