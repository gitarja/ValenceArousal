import tensorflow as tf
from Conf.Settings import EDA_N, PPG_N, Resp_N, ECG_Resp_N, ECG_N, EEG_N


class EnsembleSeparateModel(tf.keras.Model):

    def __init__(self, num_output=4, eda_length=1102, eeg_length=1330, ecg_length=48):
        super(EnsembleSeparateModel, self).__init__(self)

        # ensemble
        # EDA
        # encoder
        self.eda_en_1 = tf.keras.layers.Dense(units=256, name="eda_en_1")
        self.eda_en_2 = tf.keras.layers.Dense(units=128, name="eda_en_2")
        self.eda_en_3 = tf.keras.layers.Dense(units=64, name="eda_en_3")
        self.eda_en_4 = tf.keras.layers.Dense(units=32, name="eda_en_4")
        # decoder
        self.eda_de_1 = tf.keras.layers.Dense(units=64, name="eda_de_1")
        self.eda_de_2 = tf.keras.layers.Dense(units=128, name="eda_de_2")
        self.eda_de_3 = tf.keras.layers.Dense(units=256, name="eda_de_3")
        self.eda_de_4 = tf.keras.layers.Dense(units=eda_length, name="eda_de_4", activation=None)
        # classifer
        self.eda_ar_logit = tf.keras.layers.Dense(units=num_output, name="eda_ar_logit", activation=None)
        self.eda_val_logit = tf.keras.layers.Dense(units=num_output, name="eda_val_logit", activation=None)

        # ECG
        # encoder
        self.ecg_en_1 = tf.keras.layers.Dense(units=128, name="ecg_en_1")
        self.ecg_en_2 = tf.keras.layers.Dense(units=64, name="ecg_en_2")
        self.ecg_en_3 = tf.keras.layers.Dense(units=32, name="ecg_en_3")
        # decoder
        self.ecg_de_1 = tf.keras.layers.Dense(units=64, name="ecg_de_1")
        self.ecg_de_2 = tf.keras.layers.Dense(units=128, name="ecg_de_2")
        self.ecg_de_3 = tf.keras.layers.Dense(units=ecg_length, name="ecg_de_3")
        # classifier
        self.ecg_ar_logit = tf.keras.layers.Dense(units=num_output, name="ecg_ar_logit", activation=None)
        self.ecg_val_logit = tf.keras.layers.Dense(units=num_output, name="ecg_val_logit", activation=None)

        # EEG
        # encoder
        self.eeg_en_1 = tf.keras.layers.Dense(units=256, name="eeg_en_1")
        self.eeg_en_2 = tf.keras.layers.Dense(units=128, name="eeg_en_2")
        self.eeg_en_3 = tf.keras.layers.Dense(units=64, name="eeg_en_3")
        self.eeg_en_4 = tf.keras.layers.Dense(units=32, name="eeg_en_4")
        # decoder
        self.eeg_de_1 = tf.keras.layers.Dense(units=64, name="eeg_de_1")
        self.eeg_de_2 = tf.keras.layers.Dense(units=128, name="eeg_de_2")
        self.eeg_de_3 = tf.keras.layers.Dense(units=256, name="eeg_de_3")
        self.eeg_de_4 = tf.keras.layers.Dense(units=ecg_length, name="eeg_de_4")
        # classifier
        self.eeg_ar_logit = tf.keras.layers.Dense(units=num_output, name="eeg_ar_logit", activation=None)
        self.eeg_val_logit = tf.keras.layers.Dense(units=num_output, name="eeg_val_logit", activation=None)

        # activation
        self.activation = tf.keras.layers.ELU()
        # dropout
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        # avg
        self.avg = tf.keras.layers.Average()

        # loss
        self.cross_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                  reduction=tf.keras.losses.Reduction.NONE)
        self.rs_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def forward(self, x, dense, activation=None, droput=None):
        if activation is None:
            return droput(dense(x))
        return droput(activation(dense(x)))

    def forwardEDA(self, inputs):
        # encode
        x = self.forward(inputs, self.eda_en_1, self.activation, self.dropout1)
        x = self.forward(x, self.eda_en_2, self.activation, self.dropout1)
        x = self.forward(x, self.eda_en_3, self.activation, self.dropout1)
        z = self.forward(x, self.eda_en_4, self.activation, self.dropout1)

        # decode
        x = self.forward(z, self.eda_de_1, self.activation, self.dropout1)
        x = self.forward(x, self.eda_de_2, self.activation, self.dropout1)
        x = self.forward(x, self.eda_de_3, self.activation, self.dropout1)
        x = self.eda_de_4(x)

        # classify
        ar_logit = self.ens_1_ar(z)
        val_logit = self.ens_1_val(z)
        return ar_logit, val_logit, x

    def forwardECG(self, inputs):
        # encode
        x = self.forward(inputs, self.ecg_en_1, self.activation, self.dropout1)
        x = self.forward(x, self.ecg_en_2, self.activation, self.dropout1)
        z = self.forward(x, self.ecg_en_3, self.activation, self.dropout1)

        # decode
        x = self.forward(z, self.ecg_de_1, self.activation, self.dropout1)
        x = self.forward(x, self.ecg_de_2, self.activation, self.dropout1)
        x = self.ecg_de_3(x)

        #classify
        ar_logit = self.ens_2_ar(z)
        val_logit = self.ens_2_val(z)
        return ar_logit, val_logit, x

    def forwardEEG(self, inputs):
        # encode
        x = self.forward(inputs, self.eeg_en_1, self.activation, self.dropout1)
        x = self.forward(x, self.eeg_en_2, self.activation, self.dropout1)
        x = self.forward(x, self.eeg_en_3, self.activation, self.dropout1)
        z = self.forward(x, self.eeg_en_4, self.activation, self.dropout1)

        # encode
        x = self.forward(z, self.eeg_de_1, self.activation, self.dropout1)
        x = self.forward(x, self.eeg_de_2, self.activation, self.dropout1)
        x = self.forward(x, self.eeg_de_3, self.activation, self.dropout1)
        x = self.eeg_de_4(x)

        # decode
        ar_logit = self.ens_3_ar(z)
        val_logit = self.ens_3_val(z)
        return ar_logit, val_logit, x

    def call(self, inputs, training=None, mask=None):
        # EDA
        ar_logit_eda, val_logit_eda, x_eda = self.forwardEDA(inputs)
        # PPG, ECG
        ar_logit_ecg, val_logit_ecg, x_ecg = self.forwardECG(inputs)
        # EEG
        ar_logit_eeg, val_logit_eeg, x_eeg = self.forwardEEG(inputs)

        return [ar_logit_eda, ar_logit_ecg, ar_logit_eeg], [val_logit_eda, val_logit_ecg, val_logit_eeg], [x_eda, x_ecg, x_eeg]

    def trainSMCL(self, X, y_ar, y_val, th, global_batch_size, training=False):
        # compute AR and VAL logits
        logits = self(X, training)

        ar_logit_eda, ar_logit_ecg, ar_logit_eeg = logits[0]
        val_logit_eda, val_logit_ecg, val_logit_eeg = logits[1]
        x_eda, x_ecg, x_eeg = logits[1]


        # print(z_ecgResp_val.shape)

        # compute AR loss and AR acc
        losses_ar = tf.concat([self.loss(ar_logit_eda, y_ar), self.loss(ar_logit_ecg, y_ar), self.loss(ar_logit_eeg, y_ar)],
                              axis=-1)

        p_ar = tf.math.argmin(losses_ar, axis=1)
        mask_ar = tf.one_hot(p_ar, losses_ar.shape.as_list()[1]) + 0.3
        # mask AVG
        average_mask = tf.ones_like(mask_ar) / 6.

        # compute VAL loss and VAL ACC
        losses_val = tf.concat(
            [self.loss(ar_logit_eda, y_val), self.loss(ar_logit_ecg, y_val), self.loss(val_logit_eeg, y_val)], axis=-1)

        # compute rec loss
        losses_rec = 0.33 * (self.rs_loss(X, x_eda) + self.rs_loss(X, x_ecg) + self.rs_loss(X, x_eeg))

        p_val = tf.math.argmin(losses_val, axis=1)
        mask_val = tf.one_hot(p_val, losses_val.shape.as_list()[1]) + 0.1

        final_losses_ar = tf.nn.compute_average_loss(losses_ar, sample_weight=mask_ar,
                                                     global_batch_size=global_batch_size)
        final_losses_val = tf.nn.compute_average_loss(losses_val, sample_weight=mask_val,
                                                      global_batch_size=global_batch_size)

        final_rec_loss = tf.nn.compute_average_loss(losses_val,
                                                      global_batch_size=global_batch_size)

        logits_ar = tf.concat([tf.expand_dims(ar_logit_eda, -1), tf.expand_dims(ar_logit_ecg, -1),tf.expand_dims(ar_logit_ecg, -1)], axis=-1)
        logits_val = tf.concat([tf.expand_dims(val_logit_eda, -1), tf.expand_dims(val_logit_ecg, -1), tf.expand_dims(val_logit_eeg, -1)], axis=-1)

        predictions_ar = self.avgMultiple(logits_ar)
        predictions_val = self.avgMultiple(logits_val)

        # average loss

        avg_losses_ar = tf.nn.compute_average_loss(losses_ar, sample_weight=average_mask,
                                                   global_batch_size=global_batch_size)
        avg_losses_val = tf.nn.compute_average_loss(losses_val, sample_weight=average_mask,
                                                    global_batch_size=global_batch_size)

        return final_losses_ar, final_losses_val, final_rec_loss, predictions_ar, predictions_val, avg_losses_ar, avg_losses_val

    def loss(self, y, t):
        return tf.expand_dims(self.cross_loss(t, y), -1)

    def vote(self, y, th):
        predictions = tf.cast(tf.nn.sigmoid(y) >= th, dtype=tf.float32)
        labels = [0, 1]
        zero_val = self.tfCount(predictions, 0)
        one_val = self.tfCount(predictions, 1)
        vote = tf.argmax(tf.concat([zero_val, one_val], 1), 1)
        prediction = tf.gather(labels, vote)

        return prediction

    def voteMultiple(self, y):
        predictions = tf.nn.softmax(y, axis=1)
        labels_pred = tf.argmax(predictions, axis=1)
        labels = [0, 1, 2]
        print(labels_pred.shape)
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
        count = tf.reduce_sum(as_ints, 1)

        return tf.expand_dims(count, 1)
