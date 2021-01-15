import tensorflow as tf
import math



class EnsembleStudent(tf.keras.Model):

    def __init__(self, num_output_ar=4, num_output_val=4):
        super(EnsembleStudent, self).__init__(self)


        # activation
        self.elu = tf.keras.layers.ELU()

        # classify
        self.class_1 = tf.keras.layers.Dense(units=32, name="class_1")
        self.class_2 = tf.keras.layers.Dense(units=16, name="class_2")

        # logit
        self.logit_ar = tf.keras.layers.Dense(units=num_output_ar, activation=None, name="logit_ar")
        self.logit_val = tf.keras.layers.Dense(units=num_output_val, activation=None, name="logit_val")

        #dropout
        self.dropout_1 = tf.keras.layers.Dropout(0.3)

        # loss
        self.cross_loss = tf.losses.BinaryCrossentropy(from_logits=True,
                                                       reduction=tf.keras.losses.Reduction.NONE)
        self.multi_cross_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)


    def call(self, inputs, training=None, mask=None):
        z = self.dropout_1(self.elu(self.class_1(inputs)))
        # z = self.dropout_1(self.elu(self.class_2(z)))

        z_ar = self.logit_ar(z)
        z_val = self.logit_val(z)

        return z_ar, z_val


    def train(self, X, y_ar, y_val, th, global_batch_size, training=False):
        z_ar, z_val = self.call(X, training=training)
        final_loss_ar = tf.nn.compute_average_loss(self.cross_loss(y_ar, z_ar), global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(self.cross_loss(y_val, z_val), global_batch_size=global_batch_size)
        predictions_ar = tf.cast(tf.nn.sigmoid(z_ar) >= th, dtype=tf.float32)
        predictions_val = tf.cast(tf.nn.sigmoid(z_val) >= th, dtype=tf.float32)

        final_loss = final_loss_ar + final_loss_val
        return final_loss, predictions_ar, predictions_val

    def trainM(self, X, y_ar, y_val, th, global_batch_size, training=False):
        z_ar, z_val = self.call(X, training=training)
        final_loss_ar = tf.nn.compute_average_loss(self.cross_loss(y_ar, z_ar), global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(self.multi_cross_loss(y_val, z_val), global_batch_size=global_batch_size)
        predictions_ar = tf.cast(tf.nn.sigmoid(z_ar) >= th, dtype=tf.float32)
        predictions_val = tf.argmax(tf.nn.softmax(z_val, -1), -1)

        final_loss = final_loss_ar + final_loss_val
        return final_loss, predictions_ar, predictions_val


class EnsembleStudentOneDim(tf.keras.Model):

    def __init__(self, num_output_ar=4, num_output_val=4, pretrain=True):
        super(EnsembleStudentOneDim, self).__init__(self)
        self.en_conv1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="en_conv1",
                                               padding="same", trainable=pretrain)
        self.en_conv2 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="en_conv2",
                                               padding="same", trainable=pretrain)
        self.en_conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="en_conv3",
                                               padding="same", trainable=pretrain)
        self.en_conv4 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="en_conv4",
                                               padding="same", trainable=pretrain)
        self.en_conv5 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="en_conv5",
                                               padding="same", trainable=pretrain)
        self.en_conv6 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="en_conv6",
                                               padding="same", trainable=pretrain)


        self.batch_1 = tf.keras.layers.BatchNormalization(name="batch_1")
        self.batch_2 = tf.keras.layers.BatchNormalization(name="batch_2")
        self.batch_3 = tf.keras.layers.BatchNormalization(name="batch_3")
        self.batch_4 = tf.keras.layers.BatchNormalization(name="batch_4")
        self.batch_5 = tf.keras.layers.BatchNormalization(name="batch_5")
        self.batch_6 = tf.keras.layers.BatchNormalization(name="batch_6")

        #activation
        self.elu = tf.keras.layers.ELU()

        #classify
        self.class_1 = tf.keras.layers.Dense(units=32, name="class_1")


        #logit
        self.logit_ar = tf.keras.layers.Dense(units=num_output_ar, activation=None, name="logit_ar")
        self.logit_val = tf.keras.layers.Dense(units=num_output_val, activation=None, name="logit_val")


        #flattent
        self.flat = tf.keras.layers.Flatten()

        #pool
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=3)


        #dropout
        self.dropout_1 = tf.keras.layers.Dropout(0.3)

        # loss
        # loss
        self.cross_loss = tf.losses.BinaryCrossentropy(from_logits=True,
                                                       reduction=tf.keras.losses.Reduction.NONE)
        self.multi_cross_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                        reduction=tf.keras.losses.Reduction.NONE)
        self.mean_square_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.cos_loss = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)


    def forward(self, x, dense, norm=None, activation=None):
        if norm is None:
            return activation(dense(x))
        return activation(norm(dense(x)))



    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, -1)

        #encoder
        x = self.max_pool(self.forward(x, self.en_conv1, self.batch_1, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv2, self.batch_2, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv3, self.batch_3, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv4, self.batch_4, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv5, self.batch_5, self.elu))
        z = self.max_pool(self.forward(x, self.en_conv6, self.batch_6, self.elu))


        # print(z.shape)
        z = self.flat(z)
        z = self.elu(self.class_1(z))
        z_ar = self.logit_ar(z)
        z_val = self.logit_val(z)

        return z_ar, z_val, z


    def trainM(self, X, y_ar, y_val, y_ar_t, y_val_t, z_t, th, alpha, global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)
        y_ar_t = tf.nn.sigmoid(y_ar_t)
        y_val_t = tf.nn.sigmoid(y_val_t)
        beta = 1 -alpha
        final_loss_ar = tf.nn.compute_average_loss((alpha * self.cross_loss(y_ar, z_ar)) + (beta * self.cross_loss(y_ar_t, z_ar)), global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss((alpha * self.cross_loss(y_val, z_val)) + (beta  * self.cross_loss(y_val_t, z_val)), global_batch_size=global_batch_size)
        final_loss_rec = tf.nn.compute_average_loss(self.cos_loss(z_t, z), global_batch_size=global_batch_size)
        predictions_ar = tf.cast(tf.nn.sigmoid(z_ar) >= th, dtype=tf.float32)
        predictions_val = tf.cast(tf.nn.sigmoid(z_val) >= th, dtype=tf.float32)

        final_loss = (0.5 * (final_loss_ar + final_loss_val)) + (0.5 * final_loss_rec)
        return final_loss, predictions_ar, predictions_val

    def test(self, X, y_ar, y_val, th, global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)


        final_loss_ar = tf.nn.compute_average_loss(self.cross_loss(y_ar, z_ar) , global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(self.cross_loss(y_val, z_val) , global_batch_size=global_batch_size)

        predictions_ar = tf.cast(tf.nn.sigmoid(z_ar) >= th, dtype=tf.float32)
        predictions_val = tf.cast(tf.nn.sigmoid(z_val) >= th, dtype=tf.float32)

        final_loss = final_loss_ar + final_loss_val
        return final_loss, predictions_ar, predictions_val


class EnsembleStudentOneDim_MClass(tf.keras.Model):

    def __init__(self, num_output=4, pretrain=True):
        super(EnsembleStudentOneDim_MClass, self).__init__(self)
        self.en_conv1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="en_conv1",
                                               padding="same", trainable=pretrain)
        self.en_conv2 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="en_conv2",
                                               padding="same", trainable=pretrain)
        self.en_conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="en_conv3",
                                               padding="same", trainable=pretrain)
        self.en_conv4 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="en_conv4",
                                               padding="same", trainable=pretrain)
        self.en_conv5 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="en_conv5",
                                               padding="same", trainable=pretrain)
        self.en_conv6 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="en_conv6",
                                               padding="same", trainable=pretrain)


        self.batch_1 = tf.keras.layers.BatchNormalization(name="batch_1")
        self.batch_2 = tf.keras.layers.BatchNormalization(name="batch_2")
        self.batch_3 = tf.keras.layers.BatchNormalization(name="batch_3")
        self.batch_4 = tf.keras.layers.BatchNormalization(name="batch_4")
        self.batch_5 = tf.keras.layers.BatchNormalization(name="batch_5")
        self.batch_6 = tf.keras.layers.BatchNormalization(name="batch_6")

        #activation
        self.elu = tf.keras.layers.ELU()

        #classify
        self.class_1 = tf.keras.layers.Dense(units=32, name="class_1")


        #logit
        self.ar_logit = tf.keras.layers.Dense(units=num_output, activation=None, name="ar_logit")
        self.val_logit = tf.keras.layers.Dense(units=num_output, activation=None, name="val_logit")


        #flattent
        self.flat = tf.keras.layers.Flatten()

        #pool
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=3)


        #dropout
        self.dropout_1 = tf.keras.layers.Dropout(0.15)

        # loss

        self.multi_cross_loss = tf.losses.CategoricalCrossentropy(from_logits=True,
                                                                        reduction=tf.keras.losses.Reduction.NONE)
        self.mean_square_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)


        #weight
        self.weight = tf.constant([0.8, 0.6, 0.8])


    def forward(self, x, dense, norm=None, activation=None):
        if norm is None:
            return activation(dense(x))
        return activation(norm(dense(x)))



    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, -1)

        #encoder
        x = self.max_pool(self.forward(x, self.en_conv1, self.batch_1, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv2, self.batch_2, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv3, self.batch_3, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv4, self.batch_4, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv5, self.batch_5, self.elu))
        z = self.max_pool(self.forward(x, self.en_conv6, self.batch_6, self.elu))


        # print(z.shape)
        logit = self.flat(z)
        logit = self.dropout_1(self.elu(self.class_1(logit)))
        ar_logit = self.ar_logit(logit)
        val_logit = self.val_logit(logit)

        return ar_logit, val_logit, z


    def trainM(self, X, y_ar, y_val, y_ar_t, y_val_t, z_t, T, alpha, global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)
        y_ar_t = tf.nn.softmax(y_ar_t / T, -1)
        y_val_t = tf.nn.softmax(y_val_t / T, -1)
        beta = 1 - alpha
        final_loss_ar = tf.nn.compute_average_loss((alpha * self.multi_cross_loss(y_ar, z_ar)) + (beta * self.multi_cross_loss(y_ar_t, z_ar / T, sample_weight=self.weight)), global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(
            (alpha * self.multi_cross_loss(y_val, z_ar, sample_weight=self.weight)) + (
                        beta * self.multi_cross_loss(y_val_t, z_ar / T)), global_batch_size=global_batch_size)

        prediction_ar = tf.argmax(tf.nn.softmax(z_ar, -1), -1)
        prediction_val = tf.argmax(tf.nn.softmax(z_val, -1), -1)

        final_loss = (0.5 * (final_loss_ar + final_loss_val))
        return final_loss, prediction_ar, prediction_val

    def test(self, X, y_ar, y_val,  global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)

        final_loss_ar = tf.nn.compute_average_loss(self.sparse_cross_loss(y_ar, z_ar) , global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(self.sparse_cross_loss(y_val, z_val),
                                                   global_batch_size=global_batch_size)
        prediction_ar = tf.argmax(tf.nn.softmax(z_ar, -1), -1)
        prediction_val = tf.argmax(tf.nn.softmax(z_val, -1), -1)
        final_loss = (0.5 * (final_loss_ar + final_loss_val))
        return final_loss, prediction_ar, prediction_val





