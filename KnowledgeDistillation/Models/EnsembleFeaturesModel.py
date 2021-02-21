import tensorflow as tf


class EnsembleSeparateModel(tf.keras.Model):

    def __init__(self, num_output=4, features_length=2480):
        super(EnsembleSeparateModel, self).__init__(self)

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
        self.med_ar_logit = tf.keras.layers.Dense(units=num_output, name="med_ar_logit", activation=None)
        self.med_val_logit = tf.keras.layers.Dense(units=num_output, name="med_val_logit", activation=None)

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
        self.small_ar_logit = tf.keras.layers.Dense(units=num_output, name="small_ar_logit", activation=None)
        self.small_val_logit = tf.keras.layers.Dense(units=num_output, name="small_val_logit", activation=None)

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
        self.large_ar_logit = tf.keras.layers.Dense(units=num_output, name="large_ar_logit", activation=None)
        self.large_val_logit = tf.keras.layers.Dense(units=num_output, name="large_val_logit", activation=None)

        # activation
        self.activation = tf.keras.layers.ELU()
        # dropout
        self.dropout1 = tf.keras.layers.Dropout(0.0)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        # avg
        self.avg = tf.keras.layers.Average()

        # loss
        self.cross_loss = tf.losses.BinaryCrossentropy(from_logits=True,
                                                       reduction=tf.keras.losses.Reduction.NONE)
        self.rs_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.mean_loss = tf.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)


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
    @tf.function
    def trainSMCL(self, X, y_ar, y_val, th, ar_weight, val_weight, global_batch_size, training=False):
        # compute AR and VAL logits
        logits = self.call(X, training)

        x_med, x_small, x_large = logits[2]

        # print(z_smallResp_val.shape)

        # logit mean
        logit_ar_mean = tf.reduce_mean(logits[0], 0)
        logit_val_mean = tf.reduce_mean(logits[1], 0)

        # compute AR loss
        losses_ar = self.symmtericLoss(y_ar, logit_ar_mean)
        # compute Val loss
        losses_val = self.symmtericLoss(y_val, logit_val_mean)


        # compute rec loss

        losses_rec = 0.33 * (self.rs_loss(X, x_med) + self.rs_loss(X, x_small) + self.rs_loss(X, x_large))

        final_losses_ar = tf.nn.compute_average_loss(losses_ar,
                                                     global_batch_size=global_batch_size)
        final_losses_val = tf.nn.compute_average_loss(losses_val,
                                                      global_batch_size=global_batch_size)

        final_rec_loss = tf.nn.compute_average_loss(losses_rec,
                                                    global_batch_size=global_batch_size)


        #predictions
        predictions_ar = self.avgMultiple(logit_ar_mean, th)
        predictions_val = self.avgMultiple(logit_val_mean, th)


        final_loss_train = final_losses_ar + final_losses_val + final_rec_loss
        final_loss_ori = final_losses_ar + final_losses_val

        return final_loss_train, predictions_ar, predictions_val, final_loss_ori



    def avgMultiple(self, predictions, th=0.5):

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


class EnsembleSeparateModel_MClass(tf.keras.Model):

    def __init__(self, num_output_val=3, num_output_ar=3, features_length=2480):
        super(EnsembleSeparateModel_MClass, self).__init__(self)

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
        self.multi_cross_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                  reduction=tf.keras.losses.Reduction.NONE)
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
        losses_ar = self.multi_cross_loss(y_ar, logit_ar_mean)
        losses_val = self.multi_cross_loss(y_val, logit_val_mean)

        # compute rec loss

        losses_rec = 0.33 * (self.rs_loss(X, x_small) + self.rs_loss(X, x_med) + self.rs_loss(X, x_large))


        final_losses_ar = tf.nn.compute_average_loss(losses_ar,
                                                     global_batch_size=global_batch_size)
        final_losses_val = tf.nn.compute_average_loss(losses_val,
                                                      global_batch_size=global_batch_size)

        final_rec_loss = tf.nn.compute_average_loss(losses_rec,
                                                    global_batch_size=global_batch_size)

        predictions_ar = self.avgMultiple(logit_ar_mean)
        predictions_val = self.avgMultiple(logit_val_mean)

        final_loss_train = final_losses_ar + final_losses_val + final_rec_loss


        return final_loss_train, predictions_ar, predictions_val


    def loss(self, y, t):

        return tf.expand_dims(self.multi_cross_loss(t, y), -1)

    def avgMultiple(self, predictions):
        prob = tf.nn.softmax(predictions, -1)
        labels = tf.argmax(prob, -1)

        return tf.expand_dims(labels, -1)

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
