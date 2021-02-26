import tensorflow as tf


class FeaturesSingleModel:
    def __init__(self, output_ar, output_val):
        self.output_ar = output_ar
        self.output_val = output_val
        self.model = None

    def teacherModel(self, input_tensor):
        x = input_tensor
        for u in [128, 64, 32, 16]:
            x = tf.keras.layers.Dense(units=u)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ELU()(x)

        logit_ar = tf.keras.layers.Dense(units=self.output_ar)(x)
        logit_val = tf.keras.layers.Dense(units=self.output_val)(x)
        return logit_ar, logit_val

    def createModel(self, input_tensor):
        logit_ar, logit_val = self.teacherModel(input_tensor)
        self.model = tf.keras.models.Model(input_tensor, [logit_ar, logit_val])
        return self.model

    def computeLoss(self, x, y_ar, y_val, global_batch_size):
        loss_metrics = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                          reduction=tf.keras.losses.Reduction.NONE)
        logit_ar, logit_val = self.model(x)
        loss_ar = tf.nn.compute_average_loss(loss_metrics(y_ar, logit_ar), global_batch_size=global_batch_size)
        loss_val = tf.nn.compute_average_loss(loss_metrics(y_val, logit_val), global_batch_size=global_batch_size)
        final_loss = 0.5 * (loss_ar + loss_val)

        pred_ar = tf.cast(tf.nn.sigmoid(logit_ar) >= 0.5, tf.int32)
        pred_val = tf.cast(tf.nn.sigmoid(logit_val) >= 0.5, tf.int32)

        return final_loss, loss_ar, loss_val, pred_ar, pred_val
