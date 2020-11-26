import tensorflow as tf


class EnsembleSeparateModel(tf.keras.Model):

    def __init__(self, num_output=4):
        super(EnsembleSeparateModel, self).__init__(self)

        # EDA Model
        self.eda_dense_1 = tf.keras.layers.Dense(units=128, name="eda_dense_1")
        self.eda_dense_2 = tf.keras.layers.Dense(units=256, name="eda_dense_2")
        self.eda_dense_3 = tf.keras.layers.Dense(units=512, name="eda_dense_3")
        self.eda_dense_4 = tf.keras.layers.Dense(units=1024, name="eda_dense_4")
        self.eda_logit = tf.keras.layers.Dense(units=num_output, name="eda_logit", activation=None)
        # PPG Model
        self.ppg_dense_1 = tf.keras.layers.Dense(units=32, name="ppg_dense_1")
        self.ppg_dense_2 = tf.keras.layers.Dense(units=64, name="ppg_dense_2")
        self.ppg_dense_3 = tf.keras.layers.Dense(units=128, name="ppg_dense_3")
        self.ppg_logit = tf.keras.layers.Dense(units=num_output, name="ppg_logit", activation=None)
        # Resp Model
        self.resp_dense_1 = tf.keras.layers.Dense(units=32, name="resp_dense_1")
        self.resp_dense_2 = tf.keras.layers.Dense(units=64, name="resp_dense_2")
        self.resp_dense_3 = tf.keras.layers.Dense(units=128, name="resp_dense_3")
        self.resp_logit = tf.keras.layers.Dense(units=num_output, name="resp_logit", activation=None)
        # ECG Model
        self.ecg_dense_1 = tf.keras.layers.Dense(units=32, name="ecg_dense_1")
        self.ecg_dense_2 = tf.keras.layers.Dense(units=64, name="ecg_dense_2")
        self.ecg_dense_3 = tf.keras.layers.Dense(units=128, name="ecg_dense_3")
        self.ecg_logit = tf.keras.layers.Dense(units=num_output, name="ecg_logit", activation=None)
        # ECG_Resp Model
        self.ecgResp_dense_1 = tf.keras.layers.Dense(units=32, name="ecgResp_dense_1")
        self.ecgResp_dense_2 = tf.keras.layers.Dense(units=64, name="ecgResp_dense_2")
        self.ecgResp_dense_3 = tf.keras.layers.Dense(units=128, name="ecgResp_dense_3")
        self.ecgResp_logit = tf.keras.layers.Dense(units=num_output, name="ecgResp_logit", activation=None)
        # EEG Model
        self.eeg_dense_1 = tf.keras.layers.Dense(units=128, name="eeg_dense_1")
        self.eeg_dense_2 = tf.keras.layers.Dense(units=256, name="eeg_dense_2")
        self.eeg_dense_3 = tf.keras.layers.Dense(units=512, name="eeg_dense_3")
        self.eeg_dense_4 = tf.keras.layers.Dense(units=1024, name="eeg_dense_4")
        self.eeg_logit = tf.keras.layers.Dense(units=num_output, name="eeg_logit", activation=None)

        # activation
        self.activation = tf.keras.layers.ELU()
        # dropout
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        # avg
        self.avg = tf.keras.layers.Average()

        #loss
        self.cross_loss = tf.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)


    def forward(self, x, dense, activation=None, droput=None):
        if activation is None:
            return droput(dense(x))
        return droput(activation(dense(x)))


    def call(self, inputs, training=None, mask=None):
        # eda
        z_eda = self.forwardEDA(inputs)
        # ppg
        z_ppg = self.forwardPPG(inputs)
        # resp
        z_resp = self.forwardResp(inputs)
        # ecg
        z_ecg = self.forwardECG(inputs)
        # ecg_resp
        z_ecgResp = self.forwardECGResp(inputs)
        # eeg
        z_eeg = self.forwardEEG(inputs)

        return z_eda, z_ppg, z_resp, z_ecg, z_ecgResp, z_eeg

    def forwardEDA(self, inputs):
        x_eda = self.forward(inputs, self.eda_dense_1, self.activation, self.dropout1)
        x_eda = self.forward(x_eda, self.eda_dense_2, self.activation, self.dropout1)
        x_eda = self.forward(x_eda, self.eda_dense_3, self.activation, self.dropout1)
        x_eda = self.forward(x_eda, self.eda_dense_4, self.activation, self.dropout1)
        z_eda = self.eda_logit(x_eda)
        return z_eda

    def forwardPPG(self, inputs):
        x_ppg = self.forward(inputs, self.ppg_dense_1, self.activation, self.dropout1)
        x_ppg = self.forward(x_ppg, self.ppg_dense_2, self.activation, self.dropout1)
        x_ppg = self.forward(x_ppg, self.ppg_dense_3, self.activation, self.dropout1)
        z_ppg = self.ppg_logit(x_ppg)
        return z_ppg

    def forwardResp(self, inputs):
        x_resp = self.forward(inputs, self.resp_dense_1, self.activation, self.dropout1)
        x_resp = self.forward(x_resp, self.resp_dense_2, self.activation, self.dropout1)
        x_resp = self.forward(x_resp, self.resp_dense_3, self.activation, self.dropout1)
        z_resp = self.resp_logit(x_resp)
        return z_resp

    def forwardECG(self, inputs):
        x_ecg = self.forward(inputs, self.ecg_dense_1, self.activation, self.dropout1)
        x_ecg = self.forward(x_ecg, self.ecg_dense_2, self.activation, self.dropout1)
        x_ecg = self.forward(x_ecg, self.ecg_dense_3, self.activation, self.dropout1)
        z_ecg = self.ecg_logit(x_ecg)
        return z_ecg

    def forwardECGResp(self, inputs):
        x_ecgResp = self.forward(inputs, self.ecgResp_dense_1, self.activation, self.dropout1)
        x_ecgResp = self.forward(x_ecgResp, self.ecgResp_dense_2, self.activation, self.dropout1)
        x_ecgResp = self.forward(x_ecgResp, self.ecgResp_dense_3, self.activation, self.dropout1)
        z_ecgResp = self.ecgResp_logit(x_ecgResp)
        return z_ecgResp

    def forwardEEG(self, inputs):
        x_eeg = self.forward(inputs, self.eeg_dense_1, self.activation, self.dropout1)
        x_eeg = self.forward(x_eeg, self.eeg_dense_2, self.activation, self.dropout1)
        x_eeg = self.forward(x_eeg, self.eeg_dense_3, self.activation, self.dropout1)
        x_eeg = self.forward(x_eeg, self.eeg_dense_4, self.activation, self.dropout1)
        z_eeg = self.eeg_logit(x_eeg)
        return z_eeg

    def trainSMCL(self, X, y, global_batch_size):
        z_eda, z_ppg, z_resp, z_ecg, z_ecgResp, z_eeg = self(X)
        losses = tf.concat([self.loss(z_eda, y), self.loss(z_ppg, y), self.loss(z_resp, y),
                  self.loss(z_ecg, y),
                  self.loss(z_ecgResp, y), self.loss(z_eeg, y)], axis=1)

        p = tf.math.argmin(losses, axis=1)
        mask = tf.one_hot(p, losses.shape.as_list()[1]) + 0.5
        one_mask = tf.ones_like(losses)



        return tf.nn.compute_average_loss(losses, sample_weight=mask, global_batch_size=global_batch_size), tf.nn.compute_average_loss(losses, global_batch_size=global_batch_size)


    def loss(self, X, y):
        return  tf.expand_dims(self.cross_loss(y, X), 1)


