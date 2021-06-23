import tensorflow as tf
from KnowledgeDistillation.Utils.Losses import SoftF1Loss, PCCLoss, CCCLoss, SAGRLoss


class UnitModel(tf.keras.layers.Layer):
    def __init__(self,
                 en_units=(32, 32, 16, 16),
                 num_output=4,
                 num_subjects=28,
                 features_length=2480,
                 decoder=True,
                 embedding_n=32,
                 **kwargs):
        super(UnitModel, self).__init__(**kwargs)
        self.decoder = decoder
        if len(en_units) != 4:
            raise Exception("Units length must be 4, current unit length is" + str(len(en_units)))
        # encoder
        self.en_1 = tf.keras.layers.Dense(units=en_units[0], name="en_1", activation=None)
        self.batch_en_1 = tf.keras.layers.BatchNormalization(name="batch_en_1")
        self.en_2 = tf.keras.layers.Dense(units=en_units[1], name="en_2", activation=None)
        self.batch_en_2 = tf.keras.layers.BatchNormalization(name="batch_en_2")
        self.en_3 = tf.keras.layers.Dense(units=en_units[2], name="en_3", activation=None)
        self.batch_en_3 = tf.keras.layers.BatchNormalization(name="batch_en_3")
        self.en_4 = tf.keras.layers.Dense(units=en_units[3], name="en_4", activation=None)
        self.batch_en_4 = tf.keras.layers.BatchNormalization(name="batch_en_4")

        # decoder
        self.de_1 = tf.keras.layers.Dense(units=en_units[-1], name="de_1", activation=None)
        self.batch_de_1 = tf.keras.layers.BatchNormalization(name="batch_de_1")
        self.de_2 = tf.keras.layers.Dense(units=en_units[-2], name="de_2", activation=None)
        self.batch_de_2 = tf.keras.layers.BatchNormalization(name="batch_de_2")
        self.de_3 = tf.keras.layers.Dense(units=en_units[-3], name="de_3", activation=None)
        self.batch_de_3 = tf.keras.layers.BatchNormalization(name="batch_de_3")
        self.de_4 = tf.keras.layers.Dense(units=features_length, name="de_4", activation=None)

        # Emotion recognition
        self.embed_emotion = tf.keras.layers.Dense(units=embedding_n, activation="elu")
        # logit
        self.logit_em = tf.keras.layers.Dense(units=num_output, activation=None)
        self.logit_ar = tf.keras.layers.Dense(units=1, activation=None)
        self.logit_val = tf.keras.layers.Dense(units=1, activation=None)

        # Subject classification
        self.embed_subject = tf.keras.layers.Dense(units=embedding_n, activation="elu")
        self.logit_subject = tf.keras.layers.Dense(units=num_subjects, activation=None)

        # Gender classification
        self.embed_gender = tf.keras.layers.Dense(units=embedding_n, activation="elu")
        self.logit_gender = tf.keras.layers.Dense(units=1, activation=None)

        # activation
        self.elu = tf.keras.layers.ELU()
        self.dropout = tf.keras.layers.Dropout(0.05)

    def call(self, inputs, training=None, mask=None):

        # encoder
        x = self.elu(self.batch_en_1(self.en_1(inputs)))
        x = self.elu(self.batch_en_2(self.en_2(x)))
        x = self.elu(self.batch_en_3(self.en_3(x)))
        z = self.elu(self.batch_en_4(self.en_4(x)))

        if self.decoder:
            # decoder
            x = self.elu(self.batch_de_1(self.de_1(z)))
            x = self.elu(self.batch_de_2(self.de_2(x)))
            x = self.elu(self.batch_de_3(self.de_3(x)))
            x = self.elu(self.de_4(x))

        # Emotion recognition
        z_embed = self.embed_emotion(self.dropout(z, training=training))
        z_em = self.logit_em(z_embed)
        z_ar = self.logit_ar(z_embed)
        z_val = self.logit_val(z_embed)

        # Subject classification
        z_embed = self.embed_subject(self.dropout(z, training=training))
        z_subject = self.logit_subject(z_embed)

        # Gender classification
        z_embed = self.embed_gender(self.dropout(z, training=training))
        z_gender = self.logit_gender(z_embed)

        if self.decoder:
            return z_em, z_ar, z_val, x, z, z_subject, z_gender
        return z_em, z_ar, z_val, None, z, z_subject, z_gender


class UnitModelSingle(tf.keras.layers.Layer):

    def __init__(self, en_units=(64, 128, 256, 512), num_output=4, num_subject=28, **kwargs):
        super(UnitModelSingle, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(units=en_units[0], activation="elu")

        self.dense2 = tf.keras.layers.Dense(units=en_units[1], activation="elu")

        # ar
        self.dense3_ar = tf.keras.layers.Dense(units=en_units[2], activation="elu")

        self.dense4_ar = tf.keras.layers.Dense(units=en_units[3], activation="elu")

        # val
        self.dense3_val = tf.keras.layers.Dense(units=en_units[2], activation="elu")

        self.dense4_val = tf.keras.layers.Dense(units=en_units[3], activation="elu")

        # em
        self.dense3_em = tf.keras.layers.Dense(units=en_units[2], activation="elu")

        self.dense4_em = tf.keras.layers.Dense(units=en_units[3], activation="elu")

        # Subject
        self.dense3_sub = tf.keras.layers.Dense(units=en_units[2], activation="elu")
        self.dense4_sub = tf.keras.layers.Dense(units=en_units[3], activation="elu")

        # Gender
        self.dense3_gen = tf.keras.layers.Dense(units=en_units[2], activation="elu")
        self.dense4_gen = tf.keras.layers.Dense(units=en_units[3], activation="elu")

        # logit
        self.logit_em = tf.keras.layers.Dense(units=num_output, activation=None)
        self.logit_ar = tf.keras.layers.Dense(units=1, activation=None)
        self.logit_val = tf.keras.layers.Dense(units=1, activation=None)
        self.logit_sub = tf.keras.layers.Dense(units=num_subject, activation=None)
        self.logit_gen = tf.keras.layers.Dense(units=1, activation=None)

        # dropout
        self.dropout = tf.keras.layers.Dropout(0.05)

    def call(self, inputs, training=None, mask=None):
        x = self.dropout(self.dense1(inputs))
        z = self.dropout(self.dense2(x))

        z_em = self.logit_em(self.dense4_em(self.dense3_em(z)))
        z_ar = self.logit_ar(self.dense4_ar(self.dense3_ar(z)))
        z_val = self.logit_val(self.dense4_val(self.dense3_val(z)))
        z_sub = self.logit_sub(self.dense4_sub(self.dense3_sub(z)))
        z_gen = self.logit_gen(self.dense4_gen(self.dense3_gen(z)))

        return z_em, z_ar, z_val, z_sub, z_gen, z


class EnsembleModel(tf.keras.Model):
    def __init__(self, num_output=4):
        super(EnsembleModel, self).__init__(self)
        self.unit_small = UnitModelSingle(en_units=(16, 32, 64, 128), num_output=num_output,
                                          name="unit_small")
        self.unit_medium = UnitModelSingle(en_units=(32, 32, 64, 128), num_output=num_output,
                                           name="unit_medium")
        self.unit_large = UnitModelSingle(en_units=(32, 32, 128, 256), num_output=num_output,
                                          name="unit_large")

        # avg
        self.avg = tf.keras.layers.Average()

    def call(self, inputs, training=None, mask=None):
        # small
        z_em_s, z_ar_s, z_val_s, z_sub_s, z_gen_s, z_s = self.unit_small(inputs, training=training)
        # medium
        z_em_m, z_ar_m, z_val_m, z_sub_m, z_gen_m, z_m = self.unit_medium(inputs, training=training)
        # large
        z_em_l, z_ar_l, z_val_l, z_sub_l, z_gen_l, z_l = self.unit_large(inputs, training=training)

        z_em = self.avg([z_em_s, z_em_m, z_em_l])
        z_ar = self.avg([z_ar_s, z_ar_m, z_ar_l])
        z_val = self.avg([z_val_s, z_val_m, z_val_l])
        z_sub = self.avg([z_sub_s, z_sub_m, z_sub_l])
        z_gen = self.avg([z_gen_s, z_gen_m, z_gen_l])
        z = self.avg([z_s, z_m, z_l])
        return z_em, z_ar, z_val, z, z_sub, z_gen

    def loadBaseModel(self, checkpoint_prefix):
        model = self
        checkpoint = tf.train.Checkpoint(student_model=self)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
        checkpoint.restore(manager.latest_checkpoint).expect_partial()

        return model


class EnsembleSeparateModel(tf.keras.Model):

    def __init__(self, num_output=4, num_subject=28, features_length=2480, decoder=True):
        super(EnsembleSeparateModel, self).__init__(self)
        self.decoder = decoder
        # DNN unit

        self.unit_small = UnitModel(en_units=(32, 32, 16, 16), num_output=num_output, num_subjects=num_subject, embedding_n=32,
                                    features_length=features_length, name="unit_small", decoder=decoder)
        self.unit_medium = UnitModel(en_units=(64, 32, 16, 16), num_output=num_output, num_subjects=num_subject, embedding_n=32,
                                     features_length=features_length, name="unit_medium", decoder=decoder)
        self.unit_large = UnitModel(en_units=(64, 64, 32, 16), num_output=num_output, num_subjects=num_subject, embedding_n=32,
                                    features_length=features_length, name="unit_large", decoder=decoder)
        # avg
        self.avg = tf.keras.layers.Average()

        # loss
        self.f1_loss = SoftF1Loss(reduction=tf.keras.losses.Reduction.NONE)
        self.mae_loss = tf.losses.Huber(delta=5, reduction=tf.keras.losses.Reduction.NONE)
        self.mse_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.cross_loss = tf.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.binary_cross_loss = tf.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.pcc_loss = PCCLoss(reduction=tf.keras.losses.Reduction.NONE)
        self.ccc_loss = CCCLoss(reduction=tf.keras.losses.Reduction.NONE)
        self.sagr_loss = SAGRLoss(reduction=tf.keras.losses.Reduction.NONE)

        # l2 normalize
        self.l2_norm = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    def call(self, inputs, training=None, mask=None):
        # small
        z_em_s, z_ar_s, z_val_s, x_s, z_s, z_sub_s, z_gen_s = self.unit_small(inputs, training=training)
        # medium
        z_em_m, z_ar_m, z_val_m, x_m, z_m, z_sub_m, z_gen_m = self.unit_medium(inputs, training=training)
        # large
        z_em_l, z_ar_l, z_val_l, x_l, z_l, z_sub_l, z_gen_l = self.unit_large(inputs, training=training)

        z_em = self.avg([z_em_s, z_em_m, z_em_l])
        z_ar = self.avg([z_ar_s, z_ar_m, z_ar_l])
        z_val = self.avg([z_val_s, z_val_m, z_val_l])
        z = self.avg([z_s, z_m, z_l])
        z_sub = self.avg([z_sub_s, z_sub_m, z_sub_l])
        z_gen = self.avg([z_gen_s, z_gen_m, z_gen_l])
        if self.decoder:
            x = self.avg([x_s, x_m, x_l])
            return z_em, z_ar, z_val, x, z, z_sub, z_gen
        return z_em, z_ar, z_val, None, z, z_sub, z_gen

    def predictKD(self, X):
        logits = self.call(X, training=False)

        ar_logit = tf.reduce_mean(logits[1], axis=0)
        val_logit = tf.reduce_mean(logits[2], axis=0)
        z = tf.reduce_mean(logits[4], axis=0)

        return ar_logit, val_logit, z

    @tf.function
    def train(self, X, y_em, y_ar, y_val, global_batch_size, training=True):
        # compute AR and VAL logits
        z_em, z_ar, z_val, x, _ = self.call(X, training)

        # compute emotion loss
        losses_em = self.f1_loss(y_em, tf.sigmoid(z_em))

        # compute AR loss
        losses_ar = self.symmtericLoss(y_ar, z_ar)
        # compute Val loss
        losses_val = self.symmtericLoss(y_val, z_val)

        # compute rec loss
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

    def multiTaskClassificLoss(self, z_sub, z_gen, y_sub, y_gen, global_batch_size, alpha=0.5):
        sub_loss = tf.nn.compute_average_loss(self.cross_loss(y_sub, tf.nn.softmax(z_sub)), global_batch_size=global_batch_size)
        gen_loss = tf.nn.compute_average_loss(self.binary_cross_loss(y_gen, tf.nn.sigmoid(z_gen)), global_batch_size=global_batch_size)
        return alpha * sub_loss + (1 - alpha) * gen_loss

    def reconstructLoss(self, z, y, global_batch_size):
        rec_loss = tf.nn.compute_average_loss(
            self.mse_loss(y, z),
            global_batch_size=global_batch_size)

        return rec_loss

    def latentLoss(self, z, y, global_batch_size, sample_weight=None):
        rec_loss = tf.nn.compute_average_loss(
            self.mse_loss(y, z),
            global_batch_size=global_batch_size, sample_weight=sample_weight)

        return rec_loss

    def regressionLoss(self, z_r_ar, z_r_val, y_r_ar, y_r_val, shake_params, training=True, global_batch_size=None,
                       sample_weight=None):
        if training == True:
            a = shake_params[0] / tf.reduce_sum(shake_params)
            b = shake_params[1] / tf.reduce_sum(shake_params)
            t = shake_params[2] / tf.reduce_sum(shake_params)
        else:
            a = 0.3
            b = 0.3
            t = 0.3

        mse_loss = tf.nn.compute_average_loss(self.mse_loss(y_r_ar, z_r_ar) + self.mse_loss(y_r_val, z_r_val),
                                              global_batch_size=global_batch_size, sample_weight=sample_weight)
        pcc_loss = tf.nn.compute_average_loss(
            1 - (0.5 * (self.pcc_loss(y_r_ar, z_r_ar) + self.pcc_loss(y_r_val, z_r_val))),
            global_batch_size=global_batch_size, sample_weight=sample_weight)
        ccc_loss = tf.nn.compute_average_loss(
            1 - (0.5 * (self.ccc_loss(y_r_ar, z_r_ar) + self.ccc_loss(y_r_val, z_r_val))),
            global_batch_size=global_batch_size, sample_weight=sample_weight)

        return mse_loss, (a * mse_loss) + (b * pcc_loss) + (t * ccc_loss)

    @tf.function
    def regressionDistillLoss(self, z_r_ar, z_r_val, y_r_ar, y_r_val, t_r_ar, t_r_val, shake_params, training=True,
                              global_batch_size=None):
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

        return mse_loss, (a * mse_loss) + (b * pcc_loss) + (t * ccc_loss), mask

    def classConvert(self, predictions, th=0.5):

        labels = tf.cast(tf.nn.sigmoid(predictions) >= th, tf.int32)

        return labels

    def symmtericLoss(self, t, y, alpha=6., beta=1.):
        t2 = tf.clip_by_value(t, 1e-4, 1.0)
        y2 = tf.clip_by_value(y, 1e-7, 1.0)
        loss = alpha * self.cross_loss(t, y) + beta * self.cross_loss(y2, t2)
        return loss

    def loadBaseModel(self, checkpoint_prefix):
        model = self
        checkpoint = tf.train.Checkpoint(teacher_model=self)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
        checkpoint.restore(manager.latest_checkpoint).expect_partial()

        return model

    def modelSummary(self, input_shape=(2508,)):
        x = tf.keras.layers.Input(shape=input_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        model.summary()
