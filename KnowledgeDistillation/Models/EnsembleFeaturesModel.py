import tensorflow as tf


class EnsembleSeparateModel(tf.keras.Model):

    def __init__(self, num_output=4, features_length=2480):
        super(EnsembleSeparateModel, self).__init__(self)

        # ensemble
        # EDA
        # encoder
        self.med_en_1 = tf.keras.layers.Dense(units=64, name="med_en_1")
        self.med_en_batch_1 = tf.keras.layers.BatchNormalization()
        self.med_en_2 = tf.keras.layers.Dense(units=32, name="med_en_2")
        self.med_en_batch_2 = tf.keras.layers.BatchNormalization()
        self.med_en_3 = tf.keras.layers.Dense(units=32, name="med_en_3")
        self.med_en_batch_3 = tf.keras.layers.BatchNormalization()
        self.med_en_4 = tf.keras.layers.Dense(units=16, name="med_en_4")
        self.med_en_batch_4 = tf.keras.layers.BatchNormalization()
        # decoder
        self.med_de_1 = tf.keras.layers.Dense(units=32, name="med_de_1")
        self.med_de_batch_1 = tf.keras.layers.BatchNormalization()
        self.med_de_2 = tf.keras.layers.Dense(units=32, name="med_de_2")
        self.med_de_batch_2 = tf.keras.layers.BatchNormalization()
        self.med_de_3 = tf.keras.layers.Dense(units=64, name="med_de_3")
        self.med_de_batch_3 = tf.keras.layers.BatchNormalization()
        self.med_de_4 = tf.keras.layers.Dense(units=features_length, name="med_de_4", activation=None)
        # classifer
        self.med_class1 = tf.keras.layers.Dense(units=32, name="med_de_3")
        self.med_ar_logit = tf.keras.layers.Dense(units=num_output, name="med_ar_logit", activation=None)
        self.med_val_logit = tf.keras.layers.Dense(units=num_output, name="med_val_logit", activation=None)

        # ECG
        # encoder
        self.small_en_1 = tf.keras.layers.Dense(units=64, name="small_en_1")
        self.small_en_batch_1 = tf.keras.layers.BatchNormalization()
        self.small_en_2 = tf.keras.layers.Dense(units=32, name="small_en_2")
        self.small_en_batch_2 = tf.keras.layers.BatchNormalization()
        self.small_en_3 = tf.keras.layers.Dense(units=32, name="small_en_3")
        self.small_en_batch_3 = tf.keras.layers.BatchNormalization()
        # decoder
        self.small_de_1 = tf.keras.layers.Dense(units=32, name="small_de_1")
        self.small_de_batch_1 = tf.keras.layers.BatchNormalization()
        self.small_de_2 = tf.keras.layers.Dense(units=64, name="small_de_2")
        self.small_de_batch_2 = tf.keras.layers.BatchNormalization()
        self.small_de_3 = tf.keras.layers.Dense(units=features_length, name="small_de_3")
        # classifier
        self.small_class1 = tf.keras.layers.Dense(units=32, name="small_class1")
        self.small_ar_logit = tf.keras.layers.Dense(units=num_output, name="small_ar_logit", activation=None)
        self.small_val_logit = tf.keras.layers.Dense(units=num_output, name="small_val_logit", activation=None)

        # EEG
        # encoder
        self.large_en_1 = tf.keras.layers.Dense(units=128, name="large_en_1")
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
        self.large_de_3 = tf.keras.layers.Dense(units=128, name="large_de_3")
        self.large_de_batch_3 = tf.keras.layers.BatchNormalization()
        self.large_de_4 = tf.keras.layers.Dense(units=features_length, name="large_de_4")
        # classifier
        self.large_class1 = tf.keras.layers.Dense(units=32, name="large_class1")
        self.large_ar_logit = tf.keras.layers.Dense(units=num_output, name="large_ar_logit", activation=None)
        self.large_val_logit = tf.keras.layers.Dense(units=num_output, name="large_val_logit", activation=None)

        # activation
        self.activation = tf.keras.layers.ELU()
        # dropout
        self.dropout1 = tf.keras.layers.Dropout(0.0)
        self.dropout2 = tf.keras.layers.Dropout(0.15)
        # avg
        self.avg = tf.keras.layers.Average()

        # loss
        self.cross_loss = tf.losses.BinaryCrossentropy(from_logits=True,
                                                       reduction=tf.keras.losses.Reduction.NONE, label_smoothing=0.01)
        self.rs_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def forward(self, x, dense, activation=None, droput=None, batch_norm=None):
        if activation is None:
            return droput(dense(x))
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
        z_med = self.forward(z, self.med_class1, self.activation, self.dropout2)
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
        z_small = self.forward(z, self.small_class1, self.activation, self.dropout2)
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
        z_large = self.forward(z, self.large_class1, self.activation, self.dropout2)
        ar_logit = self.large_ar_logit(z_large)
        val_logit = self.large_val_logit(z_large)
        return ar_logit, val_logit, x, z_large

    def call(self, inputs, training=None, mask=None):

        # print(med_x)
        # EDA
        ar_logit_med, val_logit_med, x_med, z_med = self.forwardMedium(inputs)
        # PPG, ECG
        ar_logit_small, val_logit_small, x_small, z_small = self.forwardSmall(inputs)
        # EEG
        ar_logit_large, val_logit_large, x_large, z_large = self.forwardLarge(inputs)

        return [ar_logit_med, ar_logit_small, ar_logit_large], [val_logit_med, val_logit_small, val_logit_large], [x_med, x_small,
                                                                                                           x_large], [
                   z_med, z_small, z_large]

    def predictKD(self, X):
        logits = self.call(X, training=False)

        ar_logit = tf.reduce_mean(logits[0], axis=0)
        val_logit = tf.reduce_mean(logits[1], axis=0)
        z = tf.reduce_mean(logits[3], axis=0)

        return ar_logit, val_logit, z

    def trainSMCL(self, X, y_ar, y_val, th, global_batch_size, training=False):
        # compute AR and VAL logits
        logits = self.call(X, training)

        ar_logit_med, ar_logit_small, ar_logit_large = logits[0]
        val_logit_med, val_logit_small, val_logit_large = logits[1]
        x_med, x_small, x_large = logits[2]

        # print(z_smallResp_val.shape)

        # compute AR loss and AR acc
        losses_ar = tf.concat(
            [self.loss(ar_logit_med, y_ar), self.loss(ar_logit_small, y_ar), self.loss(ar_logit_large, y_ar)],
            axis=-1)

        p_ar = tf.math.argmin(losses_ar, axis=1)
        mask_ar = tf.one_hot(p_ar, losses_ar.shape.as_list()[1]) + 0.3
        # mask AVG
        average_mask = tf.ones_like(mask_ar) / 3.

        # compute VAL loss and VAL ACC
        losses_val = tf.concat(
            [self.loss(val_logit_med, y_val), self.loss(val_logit_small, y_val), self.loss(val_logit_large, y_val)],
            axis=-1)

        # compute rec loss

        losses_rec = 0.33 * (self.rs_loss(X, x_med) + self.rs_loss(X, x_small) + self.rs_loss(X, x_large))

        p_val = tf.math.argmin(losses_val, axis=1)
        mask_val = tf.one_hot(p_val, losses_val.shape.as_list()[1]) + 0.3

        final_losses_ar = tf.nn.compute_average_loss(losses_ar, sample_weight=mask_ar,
                                                     global_batch_size=global_batch_size)
        final_losses_val = tf.nn.compute_average_loss(losses_val, sample_weight=mask_val,
                                                      global_batch_size=global_batch_size)

        final_rec_loss = tf.nn.compute_average_loss(losses_rec,
                                                    global_batch_size=global_batch_size)

        logits_ar = tf.concat([ar_logit_med, ar_logit_small, ar_logit_small], axis=-1)
        logits_val = tf.concat([val_logit_med, val_logit_small, val_logit_large], axis=-1)

        predictions_ar = self.vote(logits_ar, th)
        predictions_val = self.vote(logits_val, th)

        # average loss

        avg_losses_ar = tf.nn.compute_average_loss(losses_ar, sample_weight=average_mask,
                                                   global_batch_size=global_batch_size)
        avg_losses_val = tf.nn.compute_average_loss(losses_val, sample_weight=average_mask,
                                                    global_batch_size=global_batch_size)

        final_loss_train = final_losses_ar + final_losses_val + final_rec_loss
        final_loss_ori = avg_losses_ar + avg_losses_val

        return final_loss_train, predictions_ar, predictions_val, final_loss_ori

    def loss(self, y, t):
        return tf.expand_dims(self.cross_loss(t, y), -1)

    def vote(self, y, th):
        predictions = tf.cast(tf.nn.sigmoid(y) >= th, dtype=tf.float32)
        labels = [0, 1]
        # print(predictions)
        zero_val = self.tfCount(predictions, 0)
        one_val = self.tfCount(predictions, 1)
        vote = tf.argmax(tf.concat([zero_val, one_val], -1), -1)
        prediction = tf.gather(labels, vote)

        return prediction

    def voteMultiple(self, y):
        predictions = tf.nn.softmax(y, axis=1)
        labels_pred = tf.argmax(predictions, axis=1)
        labels = [0, 1, 2]
        # print(labels_pred.shape)
        zero_val = self.tfCount(labels_pred, 0)
        one_val = self.tfCount(labels_pred, 1)
        two_val = self.tfCount(labels_pred, 2)

        vote = tf.argmax(tf.concat([zero_val, one_val, two_val], 1), 1)
        prediction = tf.gather(labels, vote)

        return prediction

    def avgMultiple(self, y):
        predictions = tf.reduce_mean(y, -1)
        prob = tf.nn.softmax(predictions)
        labels = tf.argmax(prob, -1)

        return labels

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


class EnsembleSeparateModel_MClass(tf.keras.Model):

    def __init__(self, num_output=4, features_length=2480):
        super(EnsembleSeparateModel_MClass, self).__init__(self)

        # ensemble
        # EDA
        # encoder
        self.med_en_1 = tf.keras.layers.Dense(units=64, name="med_en_1")
        self.med_en_batch_1 = tf.keras.layers.BatchNormalization()
        self.med_en_2 = tf.keras.layers.Dense(units=32, name="med_en_2")
        self.med_en_batch_2 = tf.keras.layers.BatchNormalization()
        self.med_en_3 = tf.keras.layers.Dense(units=32, name="med_en_3")
        self.med_en_batch_3 = tf.keras.layers.BatchNormalization()
        self.med_en_4 = tf.keras.layers.Dense(units=16, name="med_en_4")
        self.med_en_batch_4 = tf.keras.layers.BatchNormalization()
        # decoder
        self.med_de_1 = tf.keras.layers.Dense(units=32, name="med_de_1")
        self.med_de_batch_1 = tf.keras.layers.BatchNormalization()
        self.med_de_2 = tf.keras.layers.Dense(units=32, name="med_de_2")
        self.med_de_batch_2 = tf.keras.layers.BatchNormalization()
        self.med_de_3 = tf.keras.layers.Dense(units=64, name="med_de_3")
        self.med_de_batch_3 = tf.keras.layers.BatchNormalization()
        self.med_de_4 = tf.keras.layers.Dense(units=features_length, name="med_de_4", activation=None)
        # classifer
        self.med_class1 = tf.keras.layers.Dense(units=32, name="med_de_3")
        self.med_ar_logit = tf.keras.layers.Dense(units=num_output, name="med_ar_logit", activation=None)
        self.med_val_logit = tf.keras.layers.Dense(units=num_output, name="med_val_logit", activation=None)

        # ECG
        # encoder
        self.small_en_1 = tf.keras.layers.Dense(units=64, name="small_en_1")
        self.small_en_batch_1 = tf.keras.layers.BatchNormalization()
        self.small_en_2 = tf.keras.layers.Dense(units=32, name="small_en_2")
        self.small_en_batch_2 = tf.keras.layers.BatchNormalization()
        self.small_en_3 = tf.keras.layers.Dense(units=32, name="small_en_3")
        self.small_en_batch_3 = tf.keras.layers.BatchNormalization()
        # decoder
        self.small_de_1 = tf.keras.layers.Dense(units=32, name="small_de_1")
        self.small_de_batch_1 = tf.keras.layers.BatchNormalization()
        self.small_de_2 = tf.keras.layers.Dense(units=64, name="small_de_2")
        self.small_de_batch_2 = tf.keras.layers.BatchNormalization()
        self.small_de_3 = tf.keras.layers.Dense(units=features_length, name="small_de_3")
        # classifier
        self.small_class1 = tf.keras.layers.Dense(units=32, name="small_class1")
        self.small_ar_logit = tf.keras.layers.Dense(units=num_output, name="small_ar_logit", activation=None)
        self.small_val_logit = tf.keras.layers.Dense(units=num_output, name="small_val_logit", activation=None)

        # EEG
        # encoder
        self.large_en_1 = tf.keras.layers.Dense(units=128, name="large_en_1")
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
        self.large_de_3 = tf.keras.layers.Dense(units=128, name="large_de_3")
        self.large_de_batch_3 = tf.keras.layers.BatchNormalization()
        self.large_de_4 = tf.keras.layers.Dense(units=features_length, name="large_de_4")
        # classifier
        self.large_class1 = tf.keras.layers.Dense(units=32, name="large_class1")
        self.large_ar_logit = tf.keras.layers.Dense(units=num_output, name="large_ar_logit", activation=None)
        self.large_val_logit = tf.keras.layers.Dense(units=num_output, name="large_val_logit", activation=None)

        # activation
        self.activation = tf.keras.layers.ELU()
        # dropout
        self.dropout1 = tf.keras.layers.Dropout(0.0)
        self.dropout2 = tf.keras.layers.Dropout(0.15)
        # avg
        self.avg = tf.keras.layers.Average()

        # loss
        self.multi_cross_loss = tf.losses.CategoricalCrossentropy(from_logits=True,
                                                                  reduction=tf.keras.losses.Reduction.NONE)
        self.rs_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)



    def forward(self, x, dense, activation=None, droput=None, batch_norm=None):
        if activation is None:
            return droput(dense(x))
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
        z_med = self.forward(z, self.med_class1, self.activation, self.dropout2)
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
        z_small = self.forward(z, self.small_class1, self.activation, self.dropout2)
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
        z_large = self.forward(z, self.large_class1, self.activation, self.dropout2)
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

    def train(self, X, y_ar, y_val, th, global_batch_size, training=False):
        # compute AR and VAL logits
        logits = self.call(X, training)

        ar_logit_small, ar_logit_med, ar_logit_large = logits[0]
        val_logit_small, val_logit_med, val_logit_large = logits[1]
        x_med, x_small, x_large = logits[2]

        # logit mean
        logit_ar_mean = tf.reduce_mean(logits[0], -1)
        logit_val_mean = tf.reduce_mean(logits[1], -1)

        # compute AR loss and AR acc
        losses_ar = self.multi_cross_loss(y_ar, logit_ar_mean)
        losses_val = self.multi_cross_loss(y_val, logit_val_mean)

        # compute rec loss

        losses_rec = 0.33 * (self.rs_loss(X, x_med) + self.rs_loss(X, x_small) + self.rs_loss(X, x_large))


        final_losses_ar = tf.nn.compute_average_loss(losses_ar,
                                                     global_batch_size=global_batch_size)
        final_losses_val = tf.nn.compute_average_loss(losses_val,
                                                      global_batch_size=global_batch_size)

        final_rec_loss = tf.nn.compute_average_loss(losses_rec,
                                                    global_batch_size=global_batch_size)

        predictions_ar = self.vote(logit_ar_mean, th)
        predictions_val = self.vote(logit_val_mean, th)

        final_loss_train = final_losses_ar + final_losses_val + final_rec_loss


        return final_loss_train, predictions_ar, predictions_val


    def loss(self, y, t):

        return tf.expand_dims(self.multi_cross_loss(t, y), -1)

    def vote(self, y, th):
        predictions = tf.cast(tf.nn.sigmoid(y) >= th, dtype=tf.float32)
        labels = [0, 1]
        # print(predictions)
        zero_val = self.tfCount(predictions, 0)
        one_val = self.tfCount(predictions, 1)
        vote = tf.argmax(tf.concat([zero_val, one_val], -1), -1)
        prediction = tf.gather(labels, vote)

        return prediction

    def voteMultiple(self, y):
        predictions = tf.nn.softmax(y, axis=1)
        labels_pred = tf.argmax(predictions, axis=1)
        labels = [0, 1, 2]
        # print(labels_pred.shape)
        zero_val = self.tfCount(labels_pred, 0)
        one_val = self.tfCount(labels_pred, 1)
        two_val = self.tfCount(labels_pred, 2)

        vote = tf.argmax(tf.concat([zero_val, one_val, two_val], 1), 1)
        prediction = tf.gather(labels, vote)

        return prediction

    def avgMultiple(self, y):
        predictions = tf.reduce_mean(y, -1)
        prob = tf.nn.softmax(predictions, -1)
        labels = tf.argmax(prob, -1)

        return labels

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
