import tensorflow as tf
import tensorflow_addons as tfa
from Libs.Utils import convertContrastiveLabels


class ECGEEGEncoder:
    def __init__(self, dim_head_output=128, eeg_ch=19, eeg_n=9000):
        self.dim_head_output = dim_head_output
        self.eeg_ch = eeg_ch
        self.eeg_n = eeg_n
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

    def eegFeaturesEncoder(self, input_tensor, pretrain=True):
        x = input_tensor

        # Encoder
        for u in [512, 512, 256, 256]:
            x = tf.keras.layers.Dense(units=u, trainable=pretrain)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ELU()(x)
        h_eeg = x

        # Head
        for u in [128, 64, 32]:
            x = tf.keras.layers.Dense(units=u)(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Dropout(0.15)(x)
        z_eeg = tf.keras.layers.Dense(units=self.dim_head_output)(x)

        return h_eeg, z_eeg

    def eegNet(self, input_tensor, c=19, T=128,
               dp_rate=0.5, kern_length=64, f1=8,
               d=2, f2=16, pretrain=True):
        """
          http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

          Parameters:
          c:    channels
          T:    length of EEG
          dp_rate:  drop out rate
          d:    number of spatial  features
          f1,f2:    number of temporal features
        """

        x = tf.keras.layers.Reshape((c, T, 1))(input_tensor)
        ##################################################################
        x = tf.keras.layers.Conv2D(f1, (1, kern_length), padding='same',
                                   input_shape=(c, T, 1),
                                   use_bias=False, trainable=pretrain)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.DepthwiseConv2D((c, 1), use_bias=False,
                                            depth_multiplier=d,
                                            depthwise_constraint=tf.keras.constraints.max_norm(1.), trainable=pretrain)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.AveragePooling2D((1, 4))(x)
        x = tf.keras.layers.Dropout(dp_rate)(x)

        x = tf.keras.layers.SeparableConv2D(f2, (1, 16),
                            use_bias=False, padding='same', trainable=pretrain)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.AveragePooling2D((1, 8))(x)
        x = tf.keras.layers.Dropout(dp_rate)(x)

        x = tf.keras.layers.Flatten()(x)
        h_eeg = x

        # Head
        for u in [128, 64, 32]:
            x = tf.keras.layers.Dense(units=u)(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Dropout(0.15)(x)
        z_eeg = tf.keras.layers.Dense(units=self.dim_head_output)(x)

        return h_eeg, z_eeg


    def eegEncoder1D(self, input_tensor, pretrain=True):

        # Encoder
        x = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="same", trainable=pretrain)(
            input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=5)(x)
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=10, strides=8, padding="same", trainable=pretrain)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=5)(x)
        x = tf.keras.layers.Conv1D(filters=128, kernel_size=7, strides=1, padding="same", trainable=pretrain)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="same", trainable=pretrain)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=3)(x)
        x = tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding="same", trainable=pretrain)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding="same", trainable=pretrain)(x)
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
        # h_eeg, z_eeg = self.eegEncoder1D(input_eeg, pretrain=pretrain)
        # x_eeg = tf.keras.layers.Permute((2, 1))(input_eeg)
        # h_eeg, z_eeg = self.eegNet(x_eeg, c=self.eeg_ch, T=self.eeg_n)
        h_eeg, z_eeg = self.eegFeaturesEncoder(input_eeg, pretrain=pretrain)
        self.ecg_model = tf.keras.models.Model(input_ecg, z_ecg)
        self.eeg_model = tf.keras.models.Model(input_eeg, z_eeg)
        self.ecg_encoder = tf.keras.models.Model(self.ecg_model.input, h_ecg)
        self.eeg_encoder = tf.keras.models.Model(self.eeg_model.input, h_eeg)
        return self.ecg_model, self.eeg_model, self.ecg_encoder, self.eeg_encoder

    def contrastiveLoss(self, input_ecg, input_eeg, label_ecg, label_eeg, th=45, margin=1.0,
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

    def computeAvgLoss(self, input_ecg, input_eeg, label_ecg, label_eeg, global_batch_size, th=45, margin=1.0,
                       reduction=tf.keras.losses.Reduction.NONE):
        loss_value = self.contrastiveLoss(input_ecg, input_eeg, label_ecg, label_eeg, th, margin, reduction)
        final_loss = tf.nn.compute_average_loss(loss_value, global_batch_size=global_batch_size)
        return final_loss


class ClassifyArVal_CL:
    def __init__(self, num_output, ecg_encoder, fine_tuning=True):
        self.num_output = num_output
        self.ecg_encoder = ecg_encoder
        # self.eeg_encoder = eeg_encoder
        self.model = None
        if not fine_tuning:
            for layer in self.ecg_encoder.layers:
                layer.trainable = False
            # for layer in self.eeg_encoder.layers:
            #     layer.trainable = False

    def ClassificationModel(self):
        # x = tf.keras.layers.concatenate([self.ecg_encoder.output, self.eeg_encoder.output])
        x_ar = tf.keras.layers.Dropout(0.2)(self.ecg_encoder.output)
        x_val = tf.keras.layers.Dropout(0.2)(self.ecg_encoder.output)

        # Head
        h1_ar = tf.keras.layers.ELU()(tf.keras.layers.Dense(units=32)(x_ar))
        h1_val = tf.keras.layers.ELU()(tf.keras.layers.Dense(units=32)(x_val))
        h1_ar = tf.keras.layers.Dropout(0.2)(h1_ar)
        h1_val = tf.keras.layers.Dropout(0.2)(h1_val)
        h1_ar = tf.keras.layers.Dense(units=self.num_output)(h1_ar)
        h1_val = tf.keras.layers.Dense(units=self.num_output)(h1_val)

        h2_ar = tf.keras.layers.ELU()(tf.keras.layers.Dense(units=64)(x_ar))
        h2_val = tf.keras.layers.ELU()(tf.keras.layers.Dense(units=64)(x_val))
        h2_ar = tf.keras.layers.Dropout(0.2)(h2_ar)
        h2_val = tf.keras.layers.Dropout(0.2)(h2_val)
        h2_ar = tf.keras.layers.Dense(units=self.num_output)(h2_ar)
        h2_val = tf.keras.layers.Dense(units=self.num_output)(h2_val)

        h3_ar = tf.keras.layers.ELU()(tf.keras.layers.Dense(units=128)(x_ar))
        h3_val = tf.keras.layers.ELU()(tf.keras.layers.Dense(units=128)(x_val))
        h3_ar = tf.keras.layers.Dropout(0.2)(h3_ar)
        h3_val = tf.keras.layers.Dropout(0.2)(h3_val)
        h3_ar = tf.keras.layers.Dense(units=self.num_output)(h3_ar)
        h3_val = tf.keras.layers.Dense(units=self.num_output)(h3_val)

        # Average outputs
        z_ar = tf.keras.layers.Average()([h1_ar, h2_ar, h3_ar])
        z_val = tf.keras.layers.Average()([h1_val, h2_val, h3_val])

        return z_ar, z_val

    def createModel(self):
        z_ar, z_val = self.ClassificationModel()
        model = tf.keras.models.Model(inputs=self.ecg_encoder.input, outputs=[z_ar, z_val])
        self.model = model
        return model

    def computeLoss(self, x, y_ar, y_val, global_batch_size):
        loss_metrics = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                          reduction=tf.keras.losses.Reduction.NONE)
        logit_ar, logit_val = self.model(x)
        loss_ar = tf.nn.compute_average_loss(loss_metrics(y_ar, logit_ar), global_batch_size=global_batch_size)
        loss_val = tf.nn.compute_average_loss(loss_metrics(y_val, logit_val), global_batch_size=global_batch_size)
        # final_loss = loss_ar + loss_val

        pred_ar = tf.nn.softmax(logit_ar)
        pred_val = tf.nn.softmax(logit_val)

        return loss_ar, loss_val, pred_ar, pred_val
