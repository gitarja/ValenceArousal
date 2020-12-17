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
        self.logit = tf.keras.layers.Dense(units=num_output, activation=None, name="logit")


        #flattent
        self.flat = tf.keras.layers.Flatten()

        #pool
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=3)


        #dropout
        self.dropout_1 = tf.keras.layers.Dropout(0.15)

        # loss
        # loss
        self.sparse_cross_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                       reduction=tf.keras.losses.Reduction.NONE)
        self.multi_cross_loss = tf.losses.CategoricalCrossentropy(from_logits=True,
                                                                        reduction=tf.keras.losses.Reduction.NONE)
        self.mean_square_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)


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
        logit = self.logit(logit)

        return logit, z


    def trainM(self, X, y, y_t, z_t, T, alpha, global_batch_size, training=False):
        logit, z = self.call(X, training=training)
        y_t = tf.nn.softmax(y_t / T, -1)
        beta = 1 - alpha
        final_loss = tf.nn.compute_average_loss((alpha * self.sparse_cross_loss(y, logit)) + (beta * self.multi_cross_loss(y_t, logit / T)), global_batch_size=global_batch_size)
        prediction = tf.argmax(tf.nn.softmax(logit, -1), -1)

        return final_loss, prediction

    def test(self, X, y, global_batch_size, training=False):
        logit, z = self.call(X, training=training)

        final_loss = tf.nn.compute_average_loss(self.sparse_cross_loss(y, logit) , global_batch_size=global_batch_size)
        prediction = tf.argmax(tf.nn.softmax(logit, -1), -1)

        return final_loss, prediction

class BaseStudentOneDim(tf.keras.Model):

    def __init__(self, num_output=4, ECG_N= 11125,pretrain=True):
        super(BaseStudentOneDim, self).__init__(self)

        self.en_conv1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="en_conv1",
                                               padding="same", trainable=pretrain, input_shape=(None, ECG_N))
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

        self.de_conv6 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="de_conv6",
                                               padding="same", trainable=pretrain)
        self.de_conv5 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="de_conv5",
                                               padding="same", trainable=pretrain)
        self.de_conv4 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="de_conv4",
                                               padding="same", trainable=pretrain)
        self.de_conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="de_conv3",
                                               padding="same", trainable=pretrain)
        self.de_conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="de_conv2",
                                               padding="same", trainable=pretrain)
        self.de_conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="de_conv1",
                                               padding="same", trainable=pretrain)
        self.de_conv0 = tf.keras.layers.Conv1D(filters=1, kernel_size=5, strides=1, activation=None, name="de_conv0",
                                               padding="same", trainable=pretrain)




        #activation
        self.elu = tf.keras.layers.ELU()
        self.relu = tf.keras.layers.ReLU()

        #classify
        self.class_1 = tf.keras.layers.Dense(units=64, name="class_1")
        self.class_2 = tf.keras.layers.Dense(units=128, name="class_1")

        #logit
        self.logit = tf.keras.layers.Dense(units=num_output, activation=None, name="logit_ar")


        #flattent
        self.flat = tf.keras.layers.Flatten()

        #pool
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=3)
        self.up_samp = tf.keras.layers.UpSampling1D(size=3)

        #dropout
        self.dropout_1 = tf.keras.layers.Dropout(0.3)


        self.mean_square_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.cross_loss = tf.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)


    def forward(self, x, dense, norm=None, activation=None):
        if norm is None:
            return activation(dense(x))
        return activation(norm(dense(x)))



    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, -1)

        #encoder
        z = self.encode(x)
        x = self.decode(z)


        # print(z.shape)
        # z = self.flat(z)
        # z = self.dropout_1(self.elu(self.class_1(z)))
        # z = self.dropout_1(self.elu(self.class_2(z)))
        # z = self.relu(self.logit(z))


        return x

    def encode(self, x):
        x = self.max_pool(self.forward(x, self.en_conv1, None, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv2, None, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv3, None, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv4, None, self.elu))
        x = self.max_pool(self.forward(x, self.en_conv5, None, self.elu))
        z = self.max_pool(self.forward(x, self.en_conv6, None, self.elu))
        return z

    def decode(self, x):
        x = self.up_samp(self.forward(x, self.de_conv6, None, self.elu))
        x = self.up_samp(self.forward(x, self.de_conv5, None, self.elu))
        x = self.up_samp(self.forward(x, self.de_conv4, None, self.elu))
        x = self.up_samp(self.forward(x, self.de_conv3, None, self.elu))
        x = self.up_samp(self.forward(x, self.de_conv2, None, self.elu))
        x = self.up_samp(self.forward(x, self.de_conv1, None, self.elu))
        z = self.de_conv0(x)
        return z

    # def train(self, X, y,  global_batch_size, training=False):
    #     z = self.call(X, training=training)
    #     final_loss_ar = tf.nn.compute_average_loss(self.mean_square_loss(y, z), global_batch_size=global_batch_size)
    #     return final_loss_ar


    def train(self, X, y,  global_batch_size, training=False):
        z = self.call(X, training=training)
        final_loss_ar = tf.nn.compute_average_loss(self.cross_loss(y, z), global_batch_size=global_batch_size)
        return final_loss_ar

    def extractFeatures(self, inputs):
        x = tf.expand_dims(inputs, -1)
        #encoder
        z = self.encode(x)
        z = self.flat(z)

        return z
    def loadBaseModel(self, checkpoint_prefix):
        model = self
        checkpoint = tf.train.Checkpoint(teacher_model=self)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
        checkpoint.restore(manager.latest_checkpoint)
        model.build(input_shape=(None, 11125))


        return model




