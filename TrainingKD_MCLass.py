import tensorflow as tf
from KnowledgeDistillation.Models.EnsembleDistillModel import EnsembleStudentOneDim_MClass
from KnowledgeDistillation.Models.EnsembleFeaturesModel import EnsembleSeparateModel_MClass
from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, TRAINING_RESULTS_PATH
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetch
import datetime
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    # Create 4 virtual GPU
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

cross_tower_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=3)
strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_tower_ops)

# setting
num_output_val = 2
num_output_ar = 2
initial_learning_rate = 5.5e-4
EPOCHS = 500
PRE_EPOCHS = 100
BATCH_SIZE = 128
th = 0.5
ALL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
wait = 10
EXPECTED_ECG_SIZE = (96, 96)

# setting
fold = str(sys.argv[1])
# fold = 1
#setting model
prev_val_loss = 1000
wait_i = 0
result_path = TRAINING_RESULTS_PATH + "three-class\\fold_" + str(fold) + "\\"
checkpoint_prefix = result_path + "model_student"
# tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = result_path + "tensorboard_student\\" + current_time + '/train'
test_log_dir = result_path + "tensorboard_student\\" + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# datagenerator

training_data = DATASET_PATH + "training_data_" + str(fold) + ".csv"
validation_data = DATASET_PATH + "validation_data_" + str(fold) + ".csv"
testing_data = DATASET_PATH + "test_data_" + str(fold) + ".csv"

data_fetch = DataFetch(train_file=training_data, test_file=testing_data, validation_file=validation_data,
                       ECG_N=ECG_RAW_N, KD=True, multiple=False, soft=True)
generator = data_fetch.fetch

train_generator = tf.data.Dataset.from_generator(
    lambda: generator(training_mode=0),
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([FEATURES_N]), tf.TensorShape([num_output_ar]), tf.TensorShape([num_output_val]),
                   tf.TensorShape([ECG_RAW_N])))

val_generator = tf.data.Dataset.from_generator(
    lambda: generator(training_mode=1),
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([FEATURES_N]), tf.TensorShape([num_output_ar]), tf.TensorShape([num_output_val]),
                   tf.TensorShape([ECG_RAW_N])))

test_generator = tf.data.Dataset.from_generator(
    lambda: generator(training_mode=2),
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([FEATURES_N]), tf.TensorShape([num_output_ar]), tf.TensorShape([num_output_val]),
                   tf.TensorShape([ECG_RAW_N])))

# train dataset
train_data = train_generator.shuffle(data_fetch.train_n).repeat(3).batch(ALL_BATCH_SIZE)

val_data = val_generator.batch(BATCH_SIZE)

test_data = test_generator.batch(BATCH_SIZE)

with strategy.scope():
    # model = EnsembleStudent(num_output=num_output, expected_size=EXPECTED_ECG_SIZE)

    # load pretrained model
    checkpoint_prefix_base = result_path + "model_teacher"
    teacher_model = EnsembleSeparateModel_MClass(num_output_ar=num_output_ar, num_output_val=num_output_val).loadBaseModel(checkpoint_prefix_base)
    model = EnsembleStudentOneDim_MClass(num_output_ar=num_output_ar, num_output_val=num_output_val)

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                   decay_steps=EPOCHS, decay_rate=0.95,
                                                                   staircase=True)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)
    optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)

    # ---------------------------Epoch&Loss--------------------------#
    # loss
    train_loss = tf.keras.metrics.Mean()
    vald_loss = tf.keras.metrics.Mean()

    pre_trained_loss = tf.keras.metrics.Mean()

    # accuracy
    train_ar_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
    train_val_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=1)

    vald_ar_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
    vald_val_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=1)

    # precision

    vald_ar_pre = tf.keras.metrics.Precision()
    vald_val_pre = tf.keras.metrics.Precision()

    # recall

    vald_ar_rec = tf.keras.metrics.Recall()
    vald_val_rec = tf.keras.metrics.Recall()

# Manager
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, base_model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
# checkpoint.restore(manager.latest_checkpoint)

with strategy.scope():
    def train_step(inputs, GLOBAL_BATCH_SIZE=0):
        # X = base_model.extractFeatures(inputs[-1])
        X_t = inputs[0]
        X = inputs[-1]
        # print(X)
        y_ar = tf.concat(inputs[1], -1)
        y_val = tf.concat(inputs[2], -1)

        with tf.GradientTape() as tape:
            ar_logit, val_logit, z = teacher_model.predictKD(X_t)

            final_loss, prediction_ar, prediction_val = model.trainM(X, y_ar, y_val, ar_logit, val_logit, T=2,
                                                                     alpha=0.9, global_batch_size=
                                                                     GLOBAL_BATCH_SIZE, training=True)

        # update gradient
        grads = tape.gradient(final_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_loss(final_loss)
        # accuracy
        train_ar_acc(y_ar, prediction_ar)
        train_val_acc(y_val, prediction_val)

        return final_loss


    def test_step(inputs, GLOBAL_BATCH_SIZE=0):
        X = inputs[-1]
        # X = base_model.extractFeatures(inputs[-1])
        y_ar = tf.concat(inputs[1], -1)
        y_val = tf.concat(inputs[2], -1)

        final_loss, prediction_ar, prediction_val = model.test(X, y_ar, y_val, GLOBAL_BATCH_SIZE, training=False)
        vald_loss(final_loss)

        vald_ar_acc(y_ar, prediction_ar)
        vald_val_acc(y_val, prediction_val)

        # precision
        vald_ar_pre(tf.argmax(y_ar, -1), tf.argmax(prediction_ar, -1))
        vald_val_pre(tf.argmax(y_val, -1), tf.argmax(prediction_val, -1))
        # precision
        vald_ar_rec(tf.argmax(y_ar, -1), tf.argmax(prediction_ar, -1))
        vald_val_rec(tf.argmax(y_val, -1), tf.argmax(prediction_val, -1))

        return final_loss


    def train_reset_states():
        train_loss.reset_states()
        train_ar_acc.reset_states()
        train_val_acc.reset_states()


    def vald_reset_states():
        vald_loss.reset_states()
        vald_ar_acc.reset_states()
        vald_val_acc.reset_states()

with strategy.scope():
    # `experimental_run_v2` replicates the provided computation and runs it
    # with the distributed input.

    @tf.function
    def distributed_train_step(dataset_inputs, GLOBAL_BATCH_SIZE):
        per_replica_losses = strategy.run(train_step,
                                          args=(dataset_inputs, GLOBAL_BATCH_SIZE))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


    @tf.function
    def distributed_test_step(dataset_inputs, GLOBAL_BATCH_SIZE):
        per_replica_losses = strategy.run(test_step,
                                          args=(dataset_inputs, GLOBAL_BATCH_SIZE))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


    it = 0

    for epoch in range(EPOCHS):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0

        for step, train in enumerate(train_data):
            # print(tf.reduce_max(train[0][0]))
            distributed_train_step(train, ALL_BATCH_SIZE)
            it += 1

        with train_summary_writer.as_default():
            tf.summary.scalar('Loss', train_loss.result(), step=epoch)
            tf.summary.scalar('Arousal accuracy', train_ar_acc.result(), step=epoch)
            tf.summary.scalar('Valence accuracy', train_val_acc.result(), step=epoch)

        for step, val in enumerate(val_data):
            distributed_test_step(val, data_fetch.val_n)

        with test_summary_writer.as_default():
            tf.summary.scalar('Loss', vald_loss.result(), step=epoch)
            tf.summary.scalar('Arousal accuracy', vald_ar_acc.result(), step=epoch)
            tf.summary.scalar('Valence accuracy', vald_val_acc.result(), step=epoch)
            tf.summary.scalar('Arousal precision', vald_ar_pre.result(), step=epoch)
            tf.summary.scalar('Valence precision', vald_val_pre.result(), step=epoch)
            tf.summary.scalar('Arousal recall', vald_ar_rec.result(), step=epoch)
            tf.summary.scalar('Valence recall', vald_val_rec.result(), step=epoch)

        template = (
            "epoch {} | Train_loss: {} | Val_loss: {}")
        print(template.format(epoch + 1, train_loss.result().numpy(), vald_loss.result().numpy()))

        # Save model

        if (prev_val_loss > vald_loss.result().numpy()):
            prev_val_loss = vald_loss.result().numpy()
            wait_i = 0
            manager.save()
        else:
            wait_i += 1
        if (wait_i == wait):
            break
        # reset state
        train_reset_states()
        vald_reset_states()

    print("-------------------------------------------Testing----------------------------------------------")
    for step, test in enumerate(test_data):
        distributed_test_step(test, data_fetch.test_n)
    template = (
        "Test: loss: {}, arousal acc: {}, valence acc: {}")
    print(template.format(
        vald_loss.result().numpy(), vald_ar_acc.result().numpy(), vald_val_acc.result().numpy()))

    vald_reset_states()
    print("-----------------------------------------------------------------------------------------")
