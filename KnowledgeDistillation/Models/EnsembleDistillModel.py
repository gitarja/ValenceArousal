import tensorflow as tf
import math
from KnowledgeDistillation.Layers.AttentionLayer import AttentionLayer
from KnowledgeDistillation.Utils.Losses import PCCLoss, CCCLoss, SoftF1Loss
from KnowledgeDistillation.Layers.SelfAttentionLayer import SelfAttentionLayer1D


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


class HeadClassification(tf.keras.layers.Layer):

    def __init__(self, units, num_output=4, embedding_n=32, classification=True, **kwargs):
        super(HeadClassification, self).__init__(**kwargs)
        self.classification = classification

        # self.rnn_ar = tf.keras.layers.LSTM(units=units, name="rnn_ar")
        # self.rnn_val = tf.keras.layers.LSTM(units=units, name="rnn_val")
        # self.rnn_em = tf.keras.layers.LSTM(units=units, name="rnn_em")

        self.dense_ar = tf.keras.layers.Dense(units=units, name="dense_ar", activation="elu")
        self.dense_val = tf.keras.layers.Dense(units=units, name="dense_val", activation="elu")
        self.dense_em = tf.keras.layers.Dense(units=units, name="dense_em", activation="elu")



        # attention
        self.att = AttentionLayer(name="att_ar", TIME_STEPS=15)


        #embedding
        self.embd = tf.keras.layers.Dense(units=embedding_n, activation="elu")
        # classification
        self.logit_em = tf.keras.layers.Dense(units=num_output, activation=None, name="logit_em", kernel_initializer="he_uniform")

        # regression
        self.logit_ar_r = tf.keras.layers.Dense(units=1, activation=None, name="logit_ar", kernel_initializer="he_uniform")
        self.logit_val_r = tf.keras.layers.Dense(units=1, activation=None, name="logit_val", kernel_initializer="he_uniform")

        # activation
        self.elu = tf.keras.layers.ELU()

        # flattent
        self.flat = tf.keras.layers.Flatten()

        #dropout
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()



    def call(self, inputs, training=None):
        z = self.embd(self.flat(self.att(inputs)))
        z_ar = self.elu(self.dense_ar(z))
        z_val = self.elu(self.dense_val(z))

        # z_ar = self.rnn_ar(self.att_ar(inputs))
        # z_val = self.rnn_val(self.att_ar(inputs))


        z_ar_r = self.logit_ar_r(z_ar)
        z_val_r = self.logit_val_r(z_val)

        if self.classification:
            z_em = self.elu(self.dense_em(z))
            # z_em = self.rnn_em(self.att_em(inputs))
            z_em = self.logit_em(z_em)
            return z_em, z_ar_r, z_val_r, z

        return None, z_ar_r, z_val_r, z


class EnsembleStudentOneDim(tf.keras.Model):

    def __init__(self,  num_output=4, classification=True):
        super(EnsembleStudentOneDim, self).__init__(self)
        self.classification = classification
        self.en_conv1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="en_conv1",
                                               padding="same")
        self.en_conv2 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="en_conv2",
                                               padding="same")
        self.en_conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="en_conv3",
                                               padding="same")
        self.en_conv4 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="en_conv4",
                                               padding="same")
        self.en_conv5 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="en_conv5",
                                               padding="same")
        self.en_conv6 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="en_conv6",
                                               padding="same")

        self.batch_1 = tf.keras.layers.BatchNormalization(name="batch_1")
        self.batch_2 = tf.keras.layers.BatchNormalization(name="batch_2")
        self.batch_3 = tf.keras.layers.BatchNormalization(name="batch_3")
        self.batch_4 = tf.keras.layers.BatchNormalization(name="batch_4")
        self.batch_5 = tf.keras.layers.BatchNormalization(name="batch_5")
        self.batch_6 = tf.keras.layers.BatchNormalization(name="batch_6")


        # pool
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=3)
        self.avg_pool = tf.keras.layers.AveragePooling1D(pool_size=3)

        # dropout
        self.spatial_dropout = tf.keras.layers.SpatialDropout1D(0.05)

        # head
        self.head_small = HeadClassification(units=32, classification=classification, num_output=num_output)
        self.head_medium = HeadClassification(units=64, classification=classification, num_output=num_output)
        self.head_large = HeadClassification(units=128, classification=classification, num_output=num_output)
        # activation
        self.elu = tf.keras.layers.ELU()
        # avg
        self.avg = tf.keras.layers.Average()
        # loss
        self.sparse_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                   reduction=tf.keras.losses.Reduction.NONE)
        self.cross_loss = tf.losses.CategoricalCrossentropy(from_logits=True,
                                                            reduction=tf.keras.losses.Reduction.NONE)
        self.soft_loss = SoftF1Loss(reduction=tf.keras.losses.Reduction.NONE)
        self.mae_loss = tf.losses.Huber(delta=3, reduction=tf.keras.losses.Reduction.NONE)
        self.mse_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.pcc_loss = PCCLoss(reduction=tf.keras.losses.Reduction.NONE)
        self.ccc_loss = CCCLoss(reduction=tf.keras.losses.Reduction.NONE)



    def call(self, inputs, training=None, mask=None):
        # z = inputs
        # x = tf.expand_dims(inputs, -1)

        # encoder
        x = self.max_pool(self.elu(self.batch_1(self.en_conv1(inputs))))
        # x = self.spatial_dropout(x, training=training)
        x = self.max_pool(self.elu(self.batch_2(self.en_conv2(x))))
        # x = self.spatial_dropout(x, training=training)
        x = self.max_pool(self.elu(self.batch_3(self.en_conv3(x))))
        x = self.max_pool(self.elu(self.batch_4(self.en_conv4(x))))
        x = self.max_pool(self.elu(self.batch_5(self.en_conv5(x))))
        z = self.max_pool(self.elu(self.batch_6(self.en_conv6(x))))

        # flat logit

        z_em_s, z_ar_r_s, z_val_r_s, z_s = self.head_small(z, training=training)  # small head
        z_em_m, z_ar_r_m, z_val_r_m, z_m = self.head_medium(z, training=training)  # medium head
        z_em_l, z_ar_r_l, z_val_r_l, z_l = self.head_large(z, training=training)  # large head

        # regression
        z_ar_r = self.avg([z_ar_r_s, z_ar_r_m, z_ar_r_l])
        z_val_r = self.avg([z_val_r_s, z_val_r_m, z_val_r_l])
        z = self.avg([z_s, z_m, z_l])
        if self.classification:
            z_em = self.avg([z_em_s, z_em_m, z_em_l])
            return z_em, z_ar_r, z_val_r, z

        return None, z_ar_r, z_val_r, z

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
        mse_loss = tf.nn.compute_average_loss(0.5 * (self.mae_loss(y_r_ar, z_ar_r) + self.mae_loss(y_r_val, z_val_r)),
                                              global_batch_size=global_batch_size)
        pcc_loss = tf.nn.compute_average_loss(
            1 - (0.5 * (self.pcc_loss(y_r_ar, z_ar_r) + self.pcc_loss(y_r_val, z_val_r))),
            global_batch_size=global_batch_size)

        reg_loss = 0.5 * (mse_loss + pcc_loss)

        return final_loss_ar, final_loss_val, reg_loss, z_ar_r, z_val_r

    @tf.function
    def classificationLoss(self, z, y, global_batch_size=None):

        final_loss = tf.nn.compute_average_loss(
            self.soft_loss(y, tf.nn.sigmoid(z)),
            global_batch_size=global_batch_size)

        return final_loss



    @tf.function
    def regressionLoss(self, z_r_ar, z_r_val, y_r_ar, y_r_val, shake_params,  training=True, global_batch_size=None, sample_weight=None):
        if training == True:
            a = shake_params[0] / tf.reduce_sum(shake_params)
            b = shake_params[1] / tf.reduce_sum(shake_params)
            t = shake_params[2] / tf.reduce_sum(shake_params)
        else:
            a = 0.3
            b = 0.3
            t = 0.3

        mse_loss = tf.nn.compute_average_loss(self.mae_loss(y_r_ar, z_r_ar) + self.mae_loss(y_r_val, z_r_val),
                                              global_batch_size=global_batch_size, sample_weight=sample_weight)
        pcc_loss = tf.nn.compute_average_loss(
            1 - (0.5 * (self.pcc_loss(y_r_ar, z_r_ar) + self.pcc_loss(y_r_val, z_r_val))),
            global_batch_size=global_batch_size, sample_weight=sample_weight)
        ccc_loss = tf.nn.compute_average_loss(
            1 - (0.5 * (self.ccc_loss(y_r_ar, z_r_ar) + self.ccc_loss(y_r_val, z_r_val))),
            global_batch_size=global_batch_size, sample_weight=sample_weight)

        return mse_loss, (a * mse_loss) + (b * pcc_loss) + (t * ccc_loss)

    @tf.function
    def regressionDistillLoss(self, z_r_ar, z_r_val, y_r_ar, y_r_val, t_r_ar, t_r_val, shake_params,  training=True, global_batch_size=None):
        m = 0.5
        if training == True:
            a = shake_params[0] / tf.reduce_sum(shake_params)
            b = shake_params[1] / tf.reduce_sum(shake_params)
            t = shake_params[2] / tf.reduce_sum(shake_params)
        else:
            a = 0.3
            b = 0.3
            t = 0.3

        s_t = self.mse_loss(t_r_ar, z_r_ar) + self.mse_loss(t_r_val, z_r_val)
        s_y = self.mse_loss(y_r_ar, z_r_ar) + self.mse_loss(y_r_val, z_r_val) + m

        mask_init = tf.ones_like(s_t)
        mask = tf.where(tf.less_equal(s_y, s_t), mask_init, 0.)
        mse_loss = tf.nn.compute_average_loss(s_t,
                                              global_batch_size=global_batch_size, sample_weight=mask)
        pcc_loss = tf.nn.compute_average_loss(
            1 - (0.5 * (self.pcc_loss(y_r_ar, z_r_ar) + self.pcc_loss(y_r_val, z_r_val))),
            global_batch_size=global_batch_size, sample_weight=mask)
        ccc_loss = tf.nn.compute_average_loss(
            1 - (0.5 * (self.ccc_loss(y_r_ar, z_r_ar) + self.ccc_loss(y_r_val, z_r_val))),
            global_batch_size=global_batch_size, sample_weight=mask)

        return mse_loss, (a * mse_loss) + (b * pcc_loss) + (t * ccc_loss)

    @tf.function
    def attentiveLoss(self, z, teacher, y, eps, alpha=0.5, global_batch_size=None):

        theta = 1 - tf.reduce_sum(tf.square(teacher - y)) / eps

        lreg_loss = alpha * self.mse_loss(y, z) + (1 - alpha) * theta * self.mse_loss(teacher, z)

        lreg_loss = tf.nn.compute_average_loss(lreg_loss, global_batch_size=global_batch_size)
        return lreg_loss

    @tf.function
    def predict(self, X, global_batch_size, training=False):
        _, z_val, z_ar_r = self.call(X, training=training)

        return z_val, z_ar_r

    @tf.function
    def predict_reg(self, X, training=False):
        z_ar, z_val, z_ar_r, z_val_r = self.call(X, training=training)

        return z_ar_r, z_val_r

    def symmtericLoss(self, t, y, alpha=6.0, beta=1.):
        t2 = tf.clip_by_value(t, 1e-4, 1.0)
        y2 = tf.clip_by_value(y, 1e-7, 1.0)
        loss = alpha * self.cross_loss(t, y) + beta * self.cross_loss(y2, t2)
        return loss

    def loadBaseModel(self, checkpoint_prefix):
        model = self
        checkpoint = tf.train.Checkpoint(student_model=model)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
        checkpoint.restore(manager.latest_checkpoint).expect_partial()

        return model
