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
EPOCHS = 50
BATCH_SIZE = 128
th = 0.5
ALL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
wait = 5


for fold in range(1, 6):
    prev_val_loss = 1000
    wait_i = 0
    checkpoint_prefix = CHECK_POINT_PATH + "fold"+str(fold)
    # tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = TENSORBOARD_PATH + current_time + '/train'
    test_log_dir = TENSORBOARD_PATH + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # datagenerator

    training_data = DATASET_PATH + "training_data_"+str(fold)+".csv"
    validation_data = DATASET_PATH + "validation_data_"+str(fold)+".csv"
    testing_data = DATASET_PATH + "test_data_"+str(fold)+".csv"

    data_fetch = DataFetch(train_file=training_data, test_file=testing_data, validation_file=validation_data,
                           ECG_N=ECG_RAW_N, KD=False)
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
    train_data = train_generator.shuffle(data_fetch.train_n).repeat(3).padded_batch(BATCH_SIZE, padded_shapes=(
        tf.TensorShape([FEATURES_N]), (), ()))

    val_data = val_generator.padded_batch(BATCH_SIZE, padded_shapes=(
        tf.TensorShape([FEATURES_N]), (), ()))

    test_data = test_generator.padded_batch(BATCH_SIZE, padded_shapes=(
        tf.TensorShape([FEATURES_N]), (), ()))

    with strategy.scope():
        model = EnsembleSeparateModel(num_output=num_output)

        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                       decay_steps=EPOCHS, decay_rate=0.95, staircase=True)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)
        optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)

        # ---------------------------Epoch&Loss--------------------------#
        # loss
        train_loss = tf.keras.metrics.Mean()

        vald_loss = tf.keras.metrics.Mean()

        # accuracy
        train_ar_acc = tf.keras.metrics.BinaryAccuracy()
        train_val_acc = tf.keras.metrics.BinaryAccuracy()

        vald_ar_acc = tf.keras.metrics.BinaryAccuracy()
        vald_val_acc = tf.keras.metrics.BinaryAccuracy()

        # precision
        train_ar_pre = tf.keras.metrics.Precision()
        train_val_pre = tf.keras.metrics.Precision()

        vald_ar_pre = tf.keras.metrics.Precision()
        vald_val_pre = tf.keras.metrics.Precision()

        # recall
        train_ar_rec = tf.keras.metrics.Recall()
        train_val_rec = tf.keras.metrics.Recall()

        vald_ar_rec = tf.keras.metrics.Recall()
        vald_val_rec = tf.keras.metrics.Recall()


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
               final_loss, prediction_ar, prediction_val, loss_ori = model.trainSMCL(X, y_ar, y_val,
                                                                                                            0.55,
                                                                                                            GLOBAL_BATCH_SIZE, training=True)

            # update gradient
            grads = tape_ar.gradient(final_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_loss(loss_ori)

            #accuracy
            train_ar_acc(y_ar, prediction_ar)
            train_val_acc(y_val, prediction_val)


            #precision
            train_ar_pre(y_ar, prediction_ar)
            train_val_pre(y_val, prediction_val)

            #recall
            train_ar_rec(y_ar, prediction_ar)
            train_val_rec(y_val, prediction_val)

            return final_loss


        def test_step(inputs, GLOBAL_BATCH_SIZE=0):
            X = inputs[0]
            y_ar = tf.expand_dims(inputs[1], -1)
            y_val = tf.expand_dims(inputs[2], -1)

            final_loss, prediction_ar, prediction_val, loss_ori = model.trainSMCL(X, y_ar, y_val,
                                                                                                        0.55,
                                                                                                        GLOBAL_BATCH_SIZE, training=False)
            vald_loss(final_loss)

            vald_ar_acc(y_ar, prediction_ar)
            vald_val_acc(y_val, prediction_val)

            #precision
            vald_ar_pre(y_ar, prediction_ar)
            vald_val_pre(y_val, prediction_val)
            # precision
            vald_ar_rec(y_ar, prediction_ar)
            vald_val_rec(y_val, prediction_val)



            return final_loss


        def train_reset_states():
            train_loss.reset_states()
            train_ar_acc.reset_states()
            train_val_acc.reset_states()
            # precision
            train_ar_pre.reset_states()
            train_val_pre.reset_states()

            # recall
            train_ar_rec.reset_states()
            train_val_rec.reset_states()

        def vald_reset_states():
            vald_loss.reset_states()
            vald_ar_acc.reset_states()
            vald_val_acc.reset_states()
            # precision
            vald_ar_pre.reset_states()
            vald_val_pre.reset_states()
            # precision
            vald_ar_rec.reset_states()
            vald_val_rec.reset_states()



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
                tf.summary.scalar('Loss', train_loss.result(), step=epoch)
                tf.summary.scalar('Arousal accuracy', train_ar_acc.result(), step=epoch)
                tf.summary.scalar('Valence accuracy', train_val_acc.result(), step=epoch)
                tf.summary.scalar('Arousal precision', train_ar_pre.result(), step=epoch)
                tf.summary.scalar('Valence precision', train_val_pre.result(), step=epoch)
                tf.summary.scalar('Arousal recall', train_ar_rec.result(), step=epoch)
                tf.summary.scalar('Valence recall', train_val_rec.result(), step=epoch)



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
                "epoch {} | Train_loss: {:.4f} | Val_loss: {}")
            print(template.format(epoch + 1, train_loss.result().numpy(), vald_loss.result().numpy()))


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
        "epoch {} | Test_loss: {}")
    print(template.format(epoch + 1, vald_loss.result().numpy()))

    vald_reset_states()
    print("-----------------------------------------------------------------------------------------")
