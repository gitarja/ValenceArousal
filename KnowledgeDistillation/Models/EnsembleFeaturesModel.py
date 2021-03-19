import tensorflow as tf
from KnowledgeDistillation.Utils.Losses import SoftF1Loss, PCCLoss, CCCLoss


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
        self.dropout = tf.keras.layers.Dropout(0.3)


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
        z = self.dropout(z, training=training)
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
        self.mse_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.pcc_loss = PCCLoss(reduction=tf.keras.losses.Reduction.NONE)
        self.ccc_loss = CCCLoss(reduction=tf.keras.losses.Reduction.NONE)


    def call(self, inputs, training=None, mask=None):
        #small
        z_em_s, z_ar_s, z_val_s, x_s = self.unit_small(inputs, training=training)
        # medium
        z_em_m, z_ar_m, z_val_m, x_m = self.unit_medium(inputs, training=training)
        # large
        z_em_l, z_ar_l, z_val_l, x_l = self.unit_large(inputs, training=training)

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

    def classificationLoss(self, z, y, global_batch_size):
        final_loss = tf.nn.compute_average_loss(
            self.f1_loss(y, tf.nn.sigmoid(z)),
            global_batch_size=global_batch_size)

        return final_loss

    def reconstructLoss(self, x, y,  global_batch_size):
        rec_loss = tf.nn.compute_average_loss(
            self.mse_loss(y, x),
            global_batch_size=global_batch_size)

        return rec_loss

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

