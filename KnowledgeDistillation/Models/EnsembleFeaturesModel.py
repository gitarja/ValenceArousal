import tensorflow as tf
from Conf.Settings import EDA_N, PPG_N, Resp_N, ECG_Resp_N, ECG_N, EEG_N


class EnsembleSeparateModel(tf.keras.Model):

    def __init__(self, num_output=4):
        super(EnsembleSeparateModel, self).__init__(self)

        # ensemble
        # 1
        self.ens_11 = tf.keras.layers.Dense(units=16, name="ens_11")
        self.ens_12 = tf.keras.layers.Dense(units=32, name="ens_12")
        self.ens_13 = tf.keras.layers.Dense(units=64, name="ens_13")
        self.ens_1_ar = tf.keras.layers.Dense(units=num_output, name="ens_1_ar", activation=None)
        self.ens_1_val = tf.keras.layers.Dense(units=num_output, name="ens_1_val", activation=None)

        # 2
        self.ens_21 = tf.keras.layers.Dense(units=32, name="ens_21")
        self.ens_22 = tf.keras.layers.Dense(units=64, name="ens_22")
        self.ens_23 = tf.keras.layers.Dense(units=128, name="ens_23")
        self.ens_24 = tf.keras.layers.Dense(units=256, name="ens_24")
        self.ens_2_ar = tf.keras.layers.Dense(units=num_output, name="ens_2_ar", activation=None)
        self.ens_2_val = tf.keras.layers.Dense(units=num_output, name="ens_2_val", activation=None)

        # 3
        self.ens_31 = tf.keras.layers.Dense(units=64, name="ens_31")
        self.ens_32 = tf.keras.layers.Dense(units=64, name="ens_32")
        self.ens_33 = tf.keras.layers.Dense(units=128, name="ens_33")
        self.ens_34 = tf.keras.layers.Dense(units=256, name="ens_34")
        self.ens_3_ar = tf.keras.layers.Dense(units=num_output, name="ens_3_ar", activation=None)
        self.ens_3_val = tf.keras.layers.Dense(units=num_output, name="ens_3_val", activation=None)

        # activation
        self.activation = tf.keras.layers.ELU()
        # dropout
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        # avg
        self.avg = tf.keras.layers.Average()

        # loss
        self.cross_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def forward(self, x, dense, activation=None, droput=None):
        if activation is None:
            return droput(dense(x))
        return droput(activation(dense(x)))

    def forwardEnsemble1(self, inputs):
        x = self.forward(inputs, self.ens_11, self.activation, self.dropout1)
        x = self.forward(x, self.ens_12, self.activation, self.dropout1)
        x = self.forward(x, self.ens_13, self.activation, self.dropout1)
        ar_logit = self.ens_1_ar(x)
        val_logit = self.ens_1_val(x)
        return ar_logit, val_logit

    def forwardEnsemble2(self, inputs):
        x = self.forward(inputs, self.ens_21, self.activation, self.dropout1)
        x = self.forward(x, self.ens_22, self.activation, self.dropout1)
        x = self.forward(x, self.ens_23, self.activation, self.dropout1)
        x = self.forward(x, self.ens_24, self.activation, self.dropout1)
        ar_logit = self.ens_2_ar(x)
        val_logit = self.ens_2_val(x)
        return ar_logit, val_logit

    def forwardEnsemble3(self, inputs):
        x = self.forward(inputs, self.ens_31, self.activation, self.dropout1)
        x = self.forward(x, self.ens_32, self.activation, self.dropout1)
        x = self.forward(x, self.ens_33, self.activation, self.dropout1)
        x = self.forward(x, self.ens_34, self.activation, self.dropout1)
        ar_logit = self.ens_3_ar(x)
        val_logit = self.ens_3_val(x)
        return ar_logit, val_logit

    def call(self, inputs, training=None, mask=None):
        ar_logit_1, val_logit_1 = self.forwardEnsemble1(inputs)
        ar_logit_2, val_logit_2 = self.forwardEnsemble2(inputs)
        # ar_logit_3, val_logit_3 = self.forwardEnsemble3(inputs)

        return [ar_logit_1, ar_logit_2], [val_logit_1, val_logit_2]

    def trainSMCL(self, X, y_ar, y_val, th, global_batch_size, training=False):
        # compute AR and VAL logits
        logits = self(X, training)

        ar_logit_1, ar_logit_2 = logits[0]
        val_logit_1, val_logit_2 = logits[1]

        # print(z_ecgResp_val.shape)

        # compute AR loss and AR acc
        losses_ar = tf.concat([self.loss(ar_logit_1, y_ar), self.loss(ar_logit_2, y_ar)],
                              axis=-1)

        p_ar = tf.math.argmin(losses_ar, axis=1)
        mask_ar = tf.one_hot(p_ar, losses_ar.shape.as_list()[1]) + 0.3
        # mask AVG
        average_mask = tf.ones_like(mask_ar) / 6.

        # compute VAL loss and VAL ACC
        losses_val = tf.concat(
            [self.loss(val_logit_1, y_val), self.loss(val_logit_2, y_val)], axis=-1)

        p_val = tf.math.argmin(losses_val, axis=1)
        mask_val = tf.one_hot(p_val, losses_val.shape.as_list()[1]) + 0.1

        final_losses_ar = tf.nn.compute_average_loss(losses_ar, sample_weight=mask_ar,
                                                     global_batch_size=global_batch_size)
        final_losses_val = tf.nn.compute_average_loss(losses_val, sample_weight=mask_val,
                                                      global_batch_size=global_batch_size)

        logits_ar = tf.concat([tf.expand_dims(ar_logit_1, -1), tf.expand_dims(ar_logit_2, -1)], axis=-1)
        logits_val = tf.concat([tf.expand_dims(val_logit_1, -1), tf.expand_dims(val_logit_2, -1)], axis=-1)
        predictions_ar = self.avgMultiple(logits_ar)
        predictions_val = self.avgMultiple(logits_val)

        # average loss

        avg_losses_ar = tf.nn.compute_average_loss(losses_ar, sample_weight=average_mask,
                                                   global_batch_size=global_batch_size)
        avg_losses_val = tf.nn.compute_average_loss(losses_val, sample_weight=average_mask,
                                                    global_batch_size=global_batch_size)

        return final_losses_ar, final_losses_val, predictions_ar, predictions_val, avg_losses_ar, avg_losses_val

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
