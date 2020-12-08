import tensorflow as tf
from KnowledgeDistillation.Models.EnsembleFeaturesModel import EnsembleSeparateModel
from Conf.Settings import FEATURES_N, DATASET_PATH, ECG_RAW_N, CHECK_POINT_PATH, TENSORBOARD_PATH
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetch
from Libs.Utils import valArLevelToLabels
import datetime
import os
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
num_output = 1
initial_learning_rate = 1e-3
EPOCHS = 500
BATCH_SIZE = 64
th = 0.5
ALL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
checkpoint_prefix = CHECK_POINT_PATH + "fold0"
prev_val_loss = 1000

# tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = TENSORBOARD_PATH + current_time + '/train'
test_log_dir = TENSORBOARD_PATH + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# datagenerator

training_data = DATASET_PATH + "training_data.csv"
testing_data = DATASET_PATH + "validation_data.csv"
validation_data = DATASET_PATH + "test_data.csv"

data_fetch = DataFetch(train_file=training_data, test_file=testing_data, validation_file=validation_data,
                       ECG_N=ECG_RAW_N, max_scaler="Utils\\max_scaller.joblib",
                       norm_scaler="Utils\\norm_scaller.joblib")
generator = data_fetch.fetch

train_generator = tf.data.Dataset.from_generator(
    lambda: generator(),
    output_types=(tf.float32, tf.int32, tf.int32),
    output_shapes=(tf.TensorShape([FEATURES_N]), (), ()))

val_generator = tf.data.Dataset.from_generator(
    lambda: generator(training_mode=1),
    output_types=(tf.float32, tf.int32, tf.int32),
    output_shapes=(tf.TensorShape([FEATURES_N]), (), ()))

test_generator = tf.data.Dataset.from_generator(
    lambda: generator(training_mode=2),
    output_types=(tf.float32, tf.int32, tf.int32),
    output_shapes=(tf.TensorShape([FEATURES_N]), (), ()))

# train dataset
train_data = train_generator.shuffle(data_fetch.train_n).repeat(2).padded_batch(BATCH_SIZE, padded_shapes=(
    tf.TensorShape([FEATURES_N]), (), ()))

val_data = val_generator.padded_batch(BATCH_SIZE, padded_shapes=(
    tf.TensorShape([FEATURES_N]), (), ()))

test_data = test_generator.padded_batch(BATCH_SIZE, padded_shapes=(
    tf.TensorShape([FEATURES_N]), (), ()))

with strategy.scope():
    model = EnsembleSeparateModel(num_output=num_output)

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                   decay_steps=EPOCHS, decay_rate=0.98, staircase=True)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)
    optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)

    # ---------------------------Epoch&Loss--------------------------#
    # loss
    ar_loss_metric = tf.keras.metrics.Mean()
    val_loss_metric = tf.keras.metrics.Mean()

    # accuracy
    ar_acc = tf.keras.metrics.BinaryAccuracy()
    val_acc = tf.keras.metrics.BinaryAccuracy()

    # recall
    ar_recall = tf.keras.metrics.Recall()
    val_recall = tf.keras.metrics.Recall()

    # precision
    ar_precision = tf.keras.metrics.Precision()
    val_precision = tf.keras.metrics.Precision()

    # Manager
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, teacher_model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)

with strategy.scope():
    def train_step(inputs, GLOBAL_BATCH_SIZE=0):
        X = inputs[0]
        # print(X)
        y_ar = tf.expand_dims(inputs[1], -1)
        y_val = tf.expand_dims(inputs[2], -1)

        with tf.GradientTape() as tape_ar:
            loss_ar, loss_val, loss_rec, prediction_ar, prediction_val, loss_ar_or, loss_val_or = model.trainSMCL(X, y_ar, y_val,
                                                                                                        0.55,
                                                                                                        GLOBAL_BATCH_SIZE, training=True)
            loss = loss_ar + loss_val + loss_rec

        # update gradient
        grads = tape_ar.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        ar_loss_metric(loss_ar_or)
        val_loss_metric(loss_val_or)

        ar_acc(prediction_ar, y_ar)
        val_acc(prediction_val, y_val)

        ar_precision(prediction_ar, y_ar)
        val_precision(prediction_val, y_val)

        ar_recall(prediction_ar, y_ar)
        val_recall(prediction_val, y_val)

        return loss_ar


    def test_step(inputs, GLOBAL_BATCH_SIZE=0):
        X = inputs[0]
        y_ar = tf.expand_dims(inputs[1], -1)
        y_val = tf.expand_dims(inputs[2], -1)

        loss_ar, loss_val, loss_rec, prediction_ar, prediction_val, loss_ar_or, loss_val_or = model.trainSMCL(X, y_ar, y_val,
                                                                                                    0.55,
                                                                                                    GLOBAL_BATCH_SIZE, training=False)
        ar_loss_metric(loss_ar)
        val_loss_metric(loss_val)

        ar_acc(prediction_ar, y_ar)
        val_acc(prediction_val, y_val)

        ar_precision(prediction_ar, y_ar)
        val_precision(prediction_val, y_val)

        ar_recall(prediction_ar, y_ar)
        val_recall(prediction_val, y_val)

        return loss_ar


    def reset_states():
        ar_loss_metric.reset_states()
        val_loss_metric.reset_states()
        ar_acc.reset_states()
        val_acc.reset_states()
        ar_precision.reset_states()
        val_precision.reset_states()
        ar_recall.reset_states()
        val_recall.reset_states()

with strategy.scope():
    # `experimental_run_v2` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs, GLOBAL_BATCH_SIZE):
        per_replica_losses = strategy.experimental_run_v2(train_step,
                                          args=(dataset_inputs, GLOBAL_BATCH_SIZE))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


    def distributed_test_step(dataset_inputs, GLOBAL_BATCH_SIZE):
        per_replica_losses = strategy.experimental_run_v2(test_step,
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
            tf.summary.scalar('Arousal loss', ar_loss_metric.result(), step=epoch)
            tf.summary.scalar('Arousal accuracy', ar_acc.result(), step=epoch)
            tf.summary.scalar('Valence loss', val_loss_metric.result(), step=epoch)
            tf.summary.scalar('Valence accuracy', val_acc.result(), step=epoch)

        template = (
            "Epoch {}, ar_loss: {:.4f}, val_loss:{:.4f}, ar_acc: {}, val_acc:{}, ar_prec:{}, val_prec:{}, ar_rec:{}, val_rec:{}")
        print(template.format(epoch + 1, ar_loss_metric.result().numpy(), val_loss_metric.result().numpy(),
                              ar_acc.result().numpy(), val_acc.result().numpy(), ar_precision.result().numpy(),
                              val_precision.result().numpy(), ar_recall.result().numpy(), val_recall.result().numpy()))

        reset_states()
        if (epoch + 1) % 3 == 0:
            for step, val in enumerate(val_data):
                distributed_test_step(val, data_fetch.val_n)

            with train_summary_writer.as_default():
                tf.summary.scalar('Arousal loss', ar_loss_metric.result(), step=epoch)
                tf.summary.scalar('Arousal accuracy', ar_acc.result(), step=epoch)
                tf.summary.scalar('Valence loss', val_loss_metric.result(), step=epoch)
                tf.summary.scalar('Valence accuracy', val_acc.result(), step=epoch)
            print("--------------------------------------------Validation------------------------------------------")
            template = (
                "Val: epoch {}, ar_loss: {:.4f}, val_loss:{:.4f}, ar_acc: {}, val_acc:{}, ar_prec:{}, val_prec:{}, ar_rec:{}, val_rec:{}")
            print(template.format(epoch + 1, ar_loss_metric.result().numpy(), val_loss_metric.result().numpy(),
                                  ar_acc.result().numpy(), val_acc.result().numpy(), ar_precision.result().numpy(),
                                  val_precision.result().numpy(), ar_recall.result().numpy(),
                                  val_recall.result().numpy()))

            # Save model
            val_loss = ar_loss_metric.result().numpy() + val_loss_metric.result().numpy()
            if (prev_val_loss > val_loss):
                prev_val_loss = val_loss
                manager.save()
            # reset state
            reset_states()

            print("-----------------------------------------------------------------------------------------")

print("-------------------------------------------Testing----------------------------------------------")
for step, test in enumerate(test_data):
    distributed_test_step(test, data_fetch.test_n)
template = (
    "Epoch {}, ar_loss: {:.4f}, val_loss:{:.4f}, ar_acc: {}, val_acc:{}, ar_prec:{}, val_prec:{}, ar_rec:{}, val_rec:{}")
print(template.format(epoch + 1, ar_loss_metric.result().numpy(), val_loss_metric.result().numpy(),
                      ar_acc.result().numpy(), val_acc.result().numpy(), ar_precision.result().numpy(),
                      val_precision.result().numpy(), ar_recall.result().numpy(),
                      val_recall.result().numpy()))

reset_states()
print("-----------------------------------------------------------------------------------------")
