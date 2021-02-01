import tensorflow as tf
import tensorflow_addons as tfa
from Libs.Utils import convertContrastiveLabels


# Functional API
class ECGEEGEncoder:
    def __init__(self, dim_head_output=128):
        self.dim_head_output = dim_head_output
        self.ecg_model = None
        self.eeg_model = None
        self.ecg_encoder = None
        self.eeg_encoder = None

    def ecgEncoder(self, input_tensor, pretrain=True):
        x = tf.expand_dims(input_tensor, axis=-1)

        # Encoder
        for f in [8, 16, 32]:
            x = tf.keras.layers.Conv1D(filters=f, kernel_size=5, strides=1, padding="same", trainable=pretrain)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=3)(x)
            x = tf.keras.layers.Conv1D(filters=f, kernel_size=5, strides=1, padding="same", trainable=pretrain)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=3)(x)

        h_ecg = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.2)(h_ecg)
        # x = h_ecg

        # Head
        for u in [32, 32]:
            x = tf.keras.layers.Dense(units=u)(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Dropout(0.15)(x)
        z_ecg = tf.keras.layers.Dense(units=self.dim_head_output)(x)

        return h_ecg, z_ecg

    def eegEncoder1D(self, input_tensor, pretrain=True):

        # Encoder
        x = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="same")(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=5)(x)
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=10, strides=8, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=5)(x)
        x = tf.keras.layers.Conv1D(filters=128, kernel_size=7, strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=3)(x)
        x = tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=3)(x)
        h_eeg = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.3)(h_eeg)
        # x = h_eeg

        # Head
        for u in [128, 64, 32]:
            x = tf.keras.layers.Dense(units=u)(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Dropout(0.15)(x)
        z_eeg = tf.keras.layers.Dense(units=self.dim_head_output)(x)

        return h_eeg, z_eeg

    def eegEncoder3D(self, input_tensor, pretrain=True):

        # Encoder
        x = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                   trainable=pretrain)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)
        x = tf.keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                   trainable=pretrain)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 2, 2))(x)
        x = tf.keras.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                   trainable=pretrain)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                   trainable=pretrain)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 2, 2))(x)
        h_eeg = tf.keras.layers.GlobalAveragePooling3D()(x)

        # Head
        x = h_eeg
        for u in [1024, 512]:
            x = tf.keras.layers.Dense(units=u)(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Dropout(0.15)(x)
        z_eeg = tf.keras.layers.Dense(units=self.dim_head_output)(x)

        return h_eeg, z_eeg

    def createModel(self, input_ecg, input_eeg, pretrain=True):
        h_ecg, z_ecg = self.ecgEncoder(input_ecg, pretrain=pretrain)
        # h_eeg, z_eeg = self.eegEncoder3D(input_eeg, pretrain=pretrain)
        h_eeg, z_eeg = self.eegEncoder1D(input_eeg, pretrain=pretrain)
        self.ecg_model = tf.keras.models.Model(input_ecg, z_ecg)
        self.eeg_model = tf.keras.models.Model(input_eeg, z_eeg)
        self.ecg_encoder = tf.keras.models.Model(self.ecg_model.input, h_ecg)
        self.eeg_encoder = tf.keras.models.Model(self.eeg_model.input, h_eeg)
        return self.ecg_model, self.eeg_model, self.ecg_encoder, self.eeg_encoder

    def contrastiveLoss(self, input_ecg, input_eeg, label_ecg, label_eeg, margin=1.0,
                        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
        loss_metrics = tfa.losses.ContrastiveLoss(margin=margin, reduction=reduction)
        x_ecg = tf.convert_to_tensor(input_ecg, dtype=tf.float32)
        x_eeg = tf.convert_to_tensor(input_eeg, dtype=tf.float32)
        labels = tf.map_fn(lambda x: convertContrastiveLabels(x[0], x[1], x[2], x[3]),
                           elems=(label_ecg[0], label_eeg[0], label_ecg[1], label_eeg[1]),
                           fn_output_signature=tf.int32)
        z_ecg = self.ecg_model(x_ecg)
        z_eeg = self.eeg_model(x_eeg)
        # z_ecg = tf.math.l2_normalize(z_ecg, axis=1)
        # z_eeg = tf.math.l2_normalize(z_eeg, axis=1)
        distance_z = tf.linalg.norm(z_ecg - z_eeg, axis=1)
        loss = loss_metrics(y_true=labels, y_pred=distance_z)
        return loss

    def computeAvgLoss(self, input_ecg, input_eeg, label_ecg, label_eeg, global_batch_size, margin=1.0,
                       reduction=tf.keras.losses.Reduction.NONE):
        loss_value = self.contrastiveLoss(input_ecg, input_eeg, label_ecg, label_eeg, margin, reduction)
        final_loss = tf.nn.compute_average_loss(loss_value, global_batch_size=global_batch_size)
        return final_loss
