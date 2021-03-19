import tensorflow as tf
from KnowledgeDistillation.Utils.Losses import SoftF1Loss


class UnitModel(tf.keras.layers.Layer):
    def __init__(self, en_units=(32, 32, 16, 16), num_output=4, features_length=2480, **kwargs):
        super(UnitModel, self).__init__(**kwargs)

        if len(en_units) != 4:
            raise Exception("Units length must be 4, current unit length is" + str(len(en_units)))
        #encoder
        self.en_1 = tf.keras.layers.Dense(units=en_units[0], name="en_1", activation=None)
        self.batch_en_1 = tf.keras.layers.BatchNormalization(name="batch_en_1")
        self.en_2 = tf.keras.layers.Dense(units=en_units[1], name="en_2", activation=None)
        self.batch_en_2 = tf.keras.layers.BatchNormalization(name="batch_en_2")
        self.en_3 = tf.keras.layers.Dense(units=en_units[2], name="en_3", activation=None)
        self.batch_en_3 = tf.keras.layers.BatchNormalization(name="batch_en_3")
        self.en_4 = tf.keras.layers.Dense(units=en_units[3], name="en_4", activation=None)
        self.batch_en_4 = tf.keras.layers.BatchNormalization(name="batch_en_4")

        #decoder
        self.de_1 = tf.keras.layers.Dense(units=en_units[-1], name="de_1", activation=None)
        self.batch_de_1 = tf.keras.layers.BatchNormalization(name="batch_de_1")
        self.de_2 = tf.keras.layers.Dense(units=en_units[-2], name="de_2", activation=None)
        self.batch_de_2 = tf.keras.layers.BatchNormalization(name="batch_de_2")
        self.de_3 = tf.keras.layers.Dense(units=en_units[-3], name="de_3", activation=None)
        self.batch_de_3 = tf.keras.layers.BatchNormalization(name="batch_de_3")
        self.de_4 = tf.keras.layers.Dense(units=features_length, name="de_4", activation=None)

        #logit
        self.logit_em = tf.keras.layers.Dense(units=num_output, activation=None)
        self.logit_ar = tf.keras.layers.Dense(units=1, activation=None)
        self.logit_val = tf.keras.layers.Dense(units=1, activation=None)


        #activation
        self.elu = tf.keras.layers.ELU()


    def call(self, inputs, training=None, mask=None):

        #encoder
        x = self.elu(self.batch_en_1(self.en_1(inputs)))
        x = self.elu(self.batch_en_2(self.en_2(x)))
        x = self.elu(self.batch_en_3(self.en_3(x)))
        z = self.elu(self.batch_en_4(self.en_4(x)))

        #decoder
        x = self.elu(self.batch_de_1(self.de_1(z)))
        x = self.elu(self.batch_de_2(self.de_2(x)))
        x = self.elu(self.batch_de_3(self.de_3(x)))
        x = self.elu(self.de_4(x))

        #classification
        z_em = self.logit_em(z)
        z_ar = self.logit_ar(z)
        z_val = self.logit_val(z)

        return z_em, z_ar, z_val, x





class EnsembleSeparateModel(tf.keras.Model):

    def __init__(self, num_output=4, features_length=2480):
        super(EnsembleSeparateModel, self).__init__(self)

        #DNN unit
        self.unit_small = UnitModel(en_units=(32, 32, 16, 16), num_output=num_output, features_length=features_length, name="unit_small")
        self.unit_medium = UnitModel(en_units=(64, 32, 16, 16), num_output=num_output, features_length=features_length, name="unit_medium")
        self.unit_large = UnitModel(en_units=(64, 64, 32, 16), num_output=num_output, features_length=features_length, name="unit_large")
        # avg
        self.avg = tf.keras.layers.Average()

        # loss
        self.f1_loss = SoftF1Loss(reduction=tf.keras.losses.Reduction.NONE)
        self.rs_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.mean_loss = tf.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)


    def call(self, inputs, training=None, mask=None):
        #small
        z_em_s, z_ar_s, z_val_s, x_s = self.unit_small(inputs)
        # medium
        z_em_m, z_ar_m, z_val_m, x_m = self.unit_medium(inputs)
        # large
        z_em_l, z_ar_l, z_val_l, x_l = self.unit_large(inputs)

        z_em = self.avg([z_em_s, z_em_m, z_em_l])
        z_ar = self.avg([z_ar_s, z_ar_m, z_ar_l])
        z_val = self.avg([z_val_s, z_val_m, z_val_l])
        x = self.avg([x_s, x_m, x_l])

        return z_em, z_ar, z_val, x

    def predictKD(self, X):
        logits = self.call(X, training=False)

        ar_logit = tf.reduce_mean(logits[0], axis=0)
        val_logit = tf.reduce_mean(logits[1], axis=0)
        z = tf.reduce_mean(logits[3], axis=0)

        return ar_logit, val_logit, z
    @tf.function
    def train(self, X, y_em, y_ar, y_val, global_batch_size, training=True):
        # compute AR and VAL logits
        z_em, z_ar, z_val, x = self.call(X, training)

        # compute emotion loss
        losses_em = self.f1_loss(y_em, tf.sigmoid(z_em))

        # compute AR loss
        losses_ar = self.symmtericLoss(y_ar, z_ar)
        # compute Val loss
        losses_val = self.symmtericLoss(y_val, z_val)

        #compute rec loss
        losses_rec = self.rs_loss(x, X)

        final_losses_em = tf.nn.compute_average_loss(losses_em,
                                                     global_batch_size=global_batch_size)

        final_losses_ar = tf.nn.compute_average_loss(losses_ar,
                                                     global_batch_size=global_batch_size)
        final_losses_val = tf.nn.compute_average_loss(losses_val,
                                                      global_batch_size=global_batch_size)

        final_rec_loss = tf.nn.compute_average_loss(losses_rec,
                                                    global_batch_size=global_batch_size)

        final_loss = final_losses_em + final_losses_ar + final_losses_val + final_rec_loss

        return final_loss, self.classConvert(z_em), z_ar, z_val

    def classificationLoss(self, z_em, y_em, global_batch_size):
        final_loss = tf.nn.compute_average_loss(
            self.soft_loss(y_em, tf.nn.sigmoid(z_em)),
            global_batch_size=global_batch_size)

        return final_loss

    @tf.function
    def regressionLoss(self, z_r_ar, z_r_val, y_r_ar, y_r_val, shake_params,  training=True, global_batch_size=None):
        if training == True:
            a = shake_params[0] / tf.reduce_sum(shake_params)
            b = shake_params[1] / tf.reduce_sum(shake_params)
            t = shake_params[2] / tf.reduce_sum(shake_params)
        else:
            a = 0.3
            b = 0.3
            t = 0.3

        mse_loss = tf.nn.compute_average_loss(self.mse_loss(y_r_ar, z_r_ar) + self.mse_loss(y_r_val, z_r_val),
                                              global_batch_size=global_batch_size)
        pcc_loss = tf.nn.compute_average_loss(
            1 - (0.5 * (self.pcc_loss(y_r_ar, z_r_ar) + self.pcc_loss(y_r_val, z_r_val))),
            global_batch_size=global_batch_size)
        ccc_loss = tf.nn.compute_average_loss(
            1 - (0.5 * (self.ccc_loss(y_r_ar, z_r_ar) + self.ccc_loss(y_r_val, z_r_val))),
            global_batch_size=global_batch_size)

        return mse_loss, (a * mse_loss) + (b * pcc_loss) + (t * ccc_loss)


    def classConvert(self, predictions, th=0.5):

            labels = tf.cast(tf.nn.sigmoid(predictions) >= th, tf.int32)

            return labels

    def symmtericLoss(self, t, y, alpha=6., beta=1.):
        t2 = tf.clip_by_value(t, 1e-4, 1.0)
        y2 = tf.clip_by_value(y, 1e-7, 1.0)
        loss = alpha * self.cross_loss(t, y) + beta* self.cross_loss(y2, t2)
        return loss


    def loadBaseModel(self, checkpoint_prefix):
        model = self
        checkpoint = tf.train.Checkpoint(teacher_model=self)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
        checkpoint.restore(manager.latest_checkpoint).expect_partial()

        return model

class EnsembleSeparateModel_Regression(tf.keras.Model):

    def __init__(self, num_output_val=1, num_output_ar=1, features_length=2252):
        super(EnsembleSeparateModel_Regression, self).__init__(self)

        # ensemble
        # EDA
        # encoder
        self.med_en_1 = tf.keras.layers.Dense(units=32, name="med_en_1")
        self.med_en_batch_1 = tf.keras.layers.BatchNormalization()
        self.med_en_2 = tf.keras.layers.Dense(units=32, name="med_en_2")
        self.med_en_batch_2 = tf.keras.layers.BatchNormalization()
        self.med_en_3 = tf.keras.layers.Dense(units=16, name="med_en_3")
        self.med_en_batch_3 = tf.keras.layers.BatchNormalization()
        self.med_en_4 = tf.keras.layers.Dense(units=16, name="med_en_4")
        self.med_en_batch_4 = tf.keras.layers.BatchNormalization()
        # decoder
        self.med_de_1 = tf.keras.layers.Dense(units=16, name="med_de_1")
        self.med_de_batch_1 = tf.keras.layers.BatchNormalization()
        self.med_de_2 = tf.keras.layers.Dense(units=32, name="med_de_2")
        self.med_de_batch_2 = tf.keras.layers.BatchNormalization()
        self.med_de_3 = tf.keras.layers.Dense(units=32, name="med_de_3")
        self.med_de_batch_3 = tf.keras.layers.BatchNormalization()
        self.med_de_4 = tf.keras.layers.Dense(units=features_length, name="med_de_4", activation=None)
        # classifer
        self.med_class1 = tf.keras.layers.Dense(units=32, name="med_de_3")
        self.med_ar_logit = tf.keras.layers.Dense(units=num_output_ar, name="med_ar_logit", activation=None)
        self.med_val_logit = tf.keras.layers.Dense(units=num_output_val, name="med_val_logit", activation=None)

        # ECG
        # encoder
        self.small_en_1 = tf.keras.layers.Dense(units=64, name="small_en_1")
        self.small_en_batch_1 = tf.keras.layers.BatchNormalization()
        self.small_en_2 = tf.keras.layers.Dense(units=32, name="small_en_2")
        self.small_en_batch_2 = tf.keras.layers.BatchNormalization()
        self.small_en_3 = tf.keras.layers.Dense(units=16, name="small_en_3")
        self.small_en_batch_3 = tf.keras.layers.BatchNormalization()
        # decoder
        self.small_de_1 = tf.keras.layers.Dense(units=32, name="small_de_1")
        self.small_de_batch_1 = tf.keras.layers.BatchNormalization()
        self.small_de_2 = tf.keras.layers.Dense(units=64, name="small_de_2")
        self.small_de_batch_2 = tf.keras.layers.BatchNormalization()
        self.small_de_3 = tf.keras.layers.Dense(units=features_length, name="small_de_3")
        # classifier
        self.small_class1 = tf.keras.layers.Dense(units=32, name="small_class1")
        self.small_ar_logit = tf.keras.layers.Dense(units=num_output_ar, name="small_ar_logit", activation=None)
        self.small_val_logit = tf.keras.layers.Dense(units=num_output_val, name="small_val_logit", activation=None)

        # EEG
        # encoder
        self.large_en_1 = tf.keras.layers.Dense(units=64, name="large_en_1")
        self.large_en_batch_1 = tf.keras.layers.BatchNormalization()
        self.large_en_2 = tf.keras.layers.Dense(units=64, name="large_en_2")
        self.large_en_batch_2 = tf.keras.layers.BatchNormalization()
        self.large_en_3 = tf.keras.layers.Dense(units=32, name="large_en_3")
        self.large_en_batch_3 = tf.keras.layers.BatchNormalization()
        self.large_en_4 = tf.keras.layers.Dense(units=16, name="large_en_4")
        self.large_en_batch_4 = tf.keras.layers.BatchNormalization()
        # decoder
        self.large_de_1 = tf.keras.layers.Dense(units=32, name="large_de_1")
        self.large_de_batch_1 = tf.keras.layers.BatchNormalization()
        self.large_de_2 = tf.keras.layers.Dense(units=64, name="large_de_2")
        self.large_de_batch_2 = tf.keras.layers.BatchNormalization()
        self.large_de_3 = tf.keras.layers.Dense(units=64, name="large_de_3")
        self.large_de_batch_3 = tf.keras.layers.BatchNormalization()
        self.large_de_4 = tf.keras.layers.Dense(units=features_length, name="large_de_4")
        # classifier
        self.large_class1 = tf.keras.layers.Dense(units=32, name="large_class1")
        self.large_ar_logit = tf.keras.layers.Dense(units=num_output_ar, name="large_ar_logit", activation=None)
        self.large_val_logit = tf.keras.layers.Dense(units=num_output_val, name="large_val_logit", activation=None)

        # activation
        self.activation = tf.keras.layers.ELU()
        # dropout
        self.dropout1 = tf.keras.layers.Dropout(0.0)
        self.dropout2 = tf.keras.layers.Dropout(0.35)
        # avg
        self.avg = tf.keras.layers.Average()

        # loss
        self.rs_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def forward(self, x, dense, activation=None, droput=None, batch_norm=None):

        if activation is None:
            return droput(dense(x))
        if droput is None:
            return activation(dense(x))
        if batch_norm is None:
            return droput(activation(dense(x)))
        return activation(batch_norm(dense(x)))

    def forwardMedium(self, inputs):
        # encode
        x = self.forward(inputs, self.med_en_1, self.activation, batch_norm=self.med_en_batch_1)
        x = self.forward(x, self.med_en_2, self.activation, batch_norm=self.med_en_batch_2)
        x = self.forward(x, self.med_en_3, self.activation, batch_norm=self.med_en_batch_3)
        z = self.forward(x, self.med_en_4, self.activation, batch_norm=self.med_en_batch_4)

        # decode
        x = self.forward(z, self.med_de_1, self.activation, batch_norm=self.med_de_batch_1)
        x = self.forward(x, self.med_de_2, self.activation, batch_norm=self.med_de_batch_2)
        x = self.forward(x, self.med_de_3, self.activation, batch_norm=self.med_de_batch_3)
        x = self.med_de_4(x)

        # classify
        z_med = self.forward(self.dropout2(z), self.med_class1, self.activation)
        ar_logit = self.med_ar_logit(z_med)
        val_logit = self.med_val_logit(z_med)
        return ar_logit, val_logit, x, z_med

    def forwardSmall(self, inputs):
        # encode
        x = self.forward(inputs, self.small_en_1, self.activation, batch_norm=self.small_en_batch_1)
        x = self.forward(x, self.small_en_2, self.activation, batch_norm=self.small_en_batch_2)
        z = self.forward(x, self.small_en_3, self.activation, batch_norm=self.small_en_batch_3)

        # decode
        x = self.forward(z, self.small_de_1, self.activation, batch_norm=self.small_de_batch_1)
        x = self.forward(x, self.small_de_2, self.activation, batch_norm=self.small_de_batch_2)
        x = self.small_de_3(x)

        # classify
        z_small = self.forward(self.dropout2(z), self.small_class1, self.activation)
        ar_logit = self.small_ar_logit(z_small)
        val_logit = self.small_val_logit(z_small)
        return ar_logit, val_logit, x, z_small

    def forwardLarge(self, inputs):
        # encode
        x = self.forward(inputs, self.large_en_1, self.activation, batch_norm=self.large_en_batch_1)
        x = self.forward(x, self.large_en_2, self.activation, batch_norm=self.large_en_batch_2)
        x = self.forward(x, self.large_en_3, self.activation, batch_norm=self.large_en_batch_3)
        z = self.forward(x, self.large_en_4, self.activation, batch_norm=self.large_en_batch_4)

        # encode
        x = self.forward(z, self.large_de_1, self.activation, batch_norm=self.large_de_batch_1)
        x = self.forward(x, self.large_de_2, self.activation, batch_norm=self.large_de_batch_2)
        x = self.forward(x, self.large_de_3, self.activation, batch_norm=self.large_de_batch_3)
        x = self.large_de_4(x)

        # decode
        z_large = self.forward(self.dropout2(z), self.large_class1, self.activation)
        ar_logit = self.large_ar_logit(z_large)
        val_logit = self.large_val_logit(z_large)
        return ar_logit, val_logit, x, z_large

    def call(self, inputs, training=None, mask=None):

        # print(med_x)

        # small
        ar_logit_small, val_logit_small, x_small, z_small = self.forwardSmall(inputs)
        # med
        ar_logit_med, val_logit_med, x_med, z_med = self.forwardMedium(inputs)
        # big
        ar_logit_large, val_logit_large, x_large, z_large = self.forwardLarge(inputs)

        ar_logits = [ar_logit_small, ar_logit_med, ar_logit_large]
        val_logits = [val_logit_small, val_logit_med, val_logit_large]
        decs = [x_small, x_med, x_large]
        latents = [z_small,  z_med, z_large]
        return ar_logits, val_logits, decs, latents

    def predictKD(self, X):
        logits = self.call(X, training=False)

        ar_logit = tf.reduce_mean(logits[0], axis=0)
        val_logit = tf.reduce_mean(logits[1], axis=0)

        z = tf.reduce_mean(logits[3], axis=0)

        return ar_logit, val_logit, z

    @tf.function
    def train(self, X, y_ar, y_val, global_batch_size, training=False):
        # compute AR and VAL logits
        logits = self.call(X, training)
        x_med, x_small, x_large = logits[2]

        # logit mean
        logit_ar_mean = tf.reduce_mean(logits[0], 0)
        logit_val_mean = tf.reduce_mean(logits[1], 0)

        # compute AR loss and AR acc
        losses_ar = self.rs_loss(y_ar, logit_ar_mean)
        losses_val = self.rs_loss(y_val, logit_val_mean)

        # compute rec loss

        losses_rec = 0.33 * (self.rs_loss(X, x_small) + self.rs_loss(X, x_med) + self.rs_loss(X, x_large))


        final_losses_ar = tf.nn.compute_average_loss(losses_ar,
                                                     global_batch_size=global_batch_size)
        final_losses_val = tf.nn.compute_average_loss(losses_val,
                                                      global_batch_size=global_batch_size)

        final_rec_loss = tf.nn.compute_average_loss(losses_rec,
                                                    global_batch_size=global_batch_size)

        predictions_ar = logit_ar_mean
        predictions_val = logit_val_mean

        final_loss_train = final_losses_ar + final_losses_val + final_rec_loss


        return final_loss_train, predictions_ar, predictions_val

    def loss(self, y, t):

        return self.rs_loss(t, y)

    def tfCount(self, t, val):
        elements_equal_to_value = tf.equal(t, val)
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        count = tf.reduce_sum(as_ints, -1)

        return tf.expand_dims(count, -1)

    def loadBaseModel(self, checkpoint_prefix):
        model = self
        checkpoint = tf.train.Checkpoint(teacher_model=self)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
        checkpoint.restore(manager.latest_checkpoint).expect_partial()

        return model

