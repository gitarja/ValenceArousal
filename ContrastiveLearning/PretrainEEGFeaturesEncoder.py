import tensorflow as tf
import matplotlib.pyplot as plt
from ContrastiveLearning.Models.ContrastiveLearningModels import ECGEEGEncoder
from Conf.Settings import DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, EEG_N
from ContrastiveLearning.DataGenerator_CL import DataFetchPreTrain_CL
import os
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

cross_tower_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=1)
strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_tower_ops)

# setting
NUM_OUTPUT = 32
EPOCHS = 200
BATCH_SIZE = 256
ALL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
BATCH_DIV = 16
MARGIN = 1.0
TH = 45
WAIT = 10
initial_learning_rate = 1e-3
print("All batch size:", ALL_BATCH_SIZE)

for fold in range(1, 2):
    prev_val_loss = 1000
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    result_path = CHECK_POINT_PATH + "ContrastiveLearning\\" + current_time + "\\"
    checkpoint_prefix_ecg = result_path + "checkpoint\\ecg\\"
    checkpoint_prefix_eeg = result_path + "checkpoint\\eeg\\"

    # tensorboard
    train_log_dir = result_path + "tensorboard\\train\\"
    test_log_dir = result_path + "tensorboard\\test\\"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    training_data = DATASET_PATH + "training_data_" + str(fold) + ".csv"
    validation_data = DATASET_PATH + "validation_data_" + str(fold) + ".csv"
    testing_data = DATASET_PATH + "test_data_" + str(fold) + ".csv"

    data_fetch = DataFetchPreTrain_CL(training_data, validation_data, testing_data, ECG_RAW_N, eeg_raw=False)
    # data_fetch = DataFetchPreTrain_CL(validation_data, testing_data, testing_data, ECG_RAW_N, eeg_raw=False)
    generator = data_fetch.fetch

    train_generator_ecg = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=0, ecg_or_eeg=1),
        output_types=(tf.float32, tf.float32, tf.string),
        output_shapes=(tf.TensorShape([ECG_RAW_N]), (), ()))

    val_generator_ecg = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=1, ecg_or_eeg=1),
        output_types=(tf.float32, tf.float32, tf.string),
        output_shapes=(tf.TensorShape([ECG_RAW_N]), (), ()))

    test_generator_ecg = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=2, ecg_or_eeg=1),
        output_types=(tf.float32, tf.float32, tf.string),
        output_shapes=(tf.TensorShape([ECG_RAW_N]), (), ()))
    train_generator_eeg = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=0, ecg_or_eeg=2),
        output_types=(tf.float32, tf.float32, tf.string),
        output_shapes=(tf.TensorShape([EEG_N]), (), ()))

    val_generator_eeg = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=1, ecg_or_eeg=2),
        output_types=(tf.float32, tf.float32, tf.string),
        output_shapes=(tf.TensorShape([EEG_N]), (), ()))

    test_generator_eeg = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=2, ecg_or_eeg=2),
        output_types=(tf.float32, tf.float32, tf.string),
        output_shapes=(tf.TensorShape([EEG_N]), (), ()))

    train_data_ecg = train_generator_ecg.shuffle(data_fetch.train_n).batch(ALL_BATCH_SIZE // BATCH_DIV)
    val_data_ecg = val_generator_ecg.shuffle(data_fetch.val_n).batch(ALL_BATCH_SIZE // BATCH_DIV)
    test_data_ecg = test_generator_ecg.batch(ALL_BATCH_SIZE)
    train_data_eeg = train_generator_eeg.shuffle(data_fetch.train_n).batch(ALL_BATCH_SIZE)
    val_data_eeg = val_generator_eeg.shuffle(data_fetch.val_n).batch(ALL_BATCH_SIZE)
    test_data_eeg = test_generator_eeg.batch(ALL_BATCH_SIZE)

    with strategy.scope():

        # load pretrained model
        # checkpoint_prefix_base = CHECK_POINT_PATH + "fold_M" + str(fold)

        CL = ECGEEGEncoder(dim_head_output=NUM_OUTPUT)
        input_ecg = tf.keras.layers.Input(shape=(ECG_RAW_N,))
        input_eeg = tf.keras.layers.Input(shape=(EEG_N,))
        CL.createModel(input_ecg, input_eeg)
        CL.ecg_model.summary()
        CL.eeg_model.summary()

        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                       decay_steps=EPOCHS, decay_rate=0.95,
                                                                       staircase=True)
        # learning_rate = initial_learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # ---------------------------Epoch&Loss--------------------------#
        # loss
        train_loss = tf.keras.metrics.Mean()
        vald_loss = tf.keras.metrics.Mean()

    # Manager
    checkpoint_ecg = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, base_model=CL.ecg_encoder)
    manager_ecg = tf.train.CheckpointManager(checkpoint_ecg, checkpoint_prefix_ecg, max_to_keep=5)
    checkpoint_ecg.restore(manager_ecg.latest_checkpoint)
    checkpoint_eeg = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, base_model=CL.eeg_encoder)
    manager_eeg = tf.train.CheckpointManager(checkpoint_eeg, checkpoint_prefix_eeg, max_to_keep=5)
    checkpoint_eeg.restore(manager_eeg.latest_checkpoint)

    with strategy.scope():
        def train_step(x_ecg, x_eeg, label_ecg, label_eeg, global_batch_size, th=45, margin=1.0):
            with tf.GradientTape() as tape:
                final_loss = CL.computeAvgLoss(x_ecg, x_eeg, label_ecg, label_eeg, global_batch_size, th, margin)

            # update gradient
            grads = tape.gradient(final_loss, CL.ecg_model.trainable_variables + CL.eeg_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, CL.ecg_model.trainable_variables + CL.eeg_model.trainable_variables))

            # update_weights = CL.ecg_model.trainable_variables + CL.eeg_model.trainable_variables
            # grads = tape.gradient(final_loss, update_weights)
            # optimizer.apply_gradients(zip(grads, update_weights))

            train_loss(final_loss)

            return final_loss


        def test_step(x_ecg, x_eeg, label_ecg, label_eeg, global_batch_size, th=45, margin=1.0):
            final_loss = CL.computeAvgLoss(x_ecg, x_eeg, label_ecg, label_eeg, global_batch_size, th, margin)
            vald_loss(final_loss)

            return final_loss


        def train_reset_states():
            train_loss.reset_states()
            # train_ar_acc.reset_states()
            # train_val_acc.reset_states()


        def vald_reset_states():
            vald_loss.reset_states()
            # vald_ar_acc.reset_states()
            # vald_val_acc.reset_states()

    with strategy.scope():
        # `experimental_run_v2` replicates the provided computation and runs it
        # with the distributed input.

        @tf.function
        def distributed_train_step(x_ecg, x_eeg, label_ecg, label_eeg, global_batch_size, th=45, margin=1.0):
            per_replica_losses = strategy.run(train_step,
                                              args=(x_ecg, x_eeg, label_ecg, label_eeg, global_batch_size, th, margin))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)


        @tf.function
        def distributed_test_step(x_ecg, x_eeg, label_ecg, label_eeg, global_batch_size, th=45, margin=1.0):
            per_replica_losses = strategy.run(test_step,
                                              args=(x_ecg, x_eeg, label_ecg, label_eeg, global_batch_size, th, margin))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)


        train_loss_history = []
        val_loss_history = []

        for epoch in range(EPOCHS):
            # Train Loop
            for train_ecg in train_data_ecg:
                train_ecg = [tf.repeat(d, repeats=BATCH_DIV, axis=0) for d in train_ecg]
                for train_eeg in train_data_eeg:
                    if len(train_ecg[0]) == len(train_eeg[0]):
                        distributed_train_step(train_ecg[0], train_eeg[0], train_ecg[1:3],
                                               train_eeg[1:3], global_batch_size=ALL_BATCH_SIZE, th=TH, margin=MARGIN)

            # Also use validation data
            for val_ecg in val_data_ecg:
                val_ecg = [tf.repeat(d, repeats=BATCH_DIV, axis=0) for d in val_ecg]
                for val_eeg in val_data_eeg:
                    if len(val_ecg[0]) == len(val_eeg[0]):
                        distributed_train_step(val_ecg[0], val_eeg[0], val_ecg[1:3],
                                               val_eeg[1:3], global_batch_size=ALL_BATCH_SIZE, th=TH, margin=MARGIN)

            with train_summary_writer.as_default():
                tf.summary.scalar('Loss', train_loss.result(), step=epoch)

            # Validation Loop
            for test_ecg in test_data_ecg:
                for test_eeg in test_data_eeg:
                    if len(test_ecg[0]) == len(test_eeg[0]):
                        distributed_test_step(test_ecg[0], test_eeg[0], test_ecg[1:3],
                                              test_eeg[1:3], global_batch_size=ALL_BATCH_SIZE, th=TH, margin=MARGIN)

            with test_summary_writer.as_default():
                tf.summary.scalar('Loss', vald_loss.result(), step=epoch)

            train_loss_history.append(train_loss.result().numpy())
            val_loss_history.append(vald_loss.result().numpy())

            template = "epoch {}/{} | Train_loss: {} | Val_loss: {}"
            print(template.format(epoch + 1, EPOCHS, train_loss.result().numpy(), vald_loss.result().numpy()))
            lr_now = optimizer._decayed_lr(tf.float32).numpy()
            print("Now learning rate:", lr_now)

            # Save model
            if prev_val_loss > vald_loss.result().numpy():
                prev_val_loss = vald_loss.result().numpy()
                wait_i = 0
                manager_ecg.save()
                manager_eeg.save()
            else:
                wait_i += 1
            if wait_i == WAIT:
                break

            # reset state
            train_reset_states()
            vald_reset_states()

    print("-------------------------------------------Testing----------------------------------------------")
    for test_ecg in test_data_ecg:
        for test_eeg in test_data_eeg:
            if len(test_ecg[0]) == len(test_eeg[0]):
                distributed_test_step(test_ecg[0], test_eeg[0], test_ecg[1:3],
                                      test_eeg[1:3], global_batch_size=ALL_BATCH_SIZE, th=TH, margin=MARGIN)
    template = "Test: loss: {}"
    print(template.format(vald_loss.result().numpy()))

    vald_reset_states()
    print("-----------------------------------------------------------------------------------------")

    # Save weights
    CL.ecg_encoder.save_weights(result_path + "ECG_encoder_weights_" + str(fold) + ".hdf5")
    CL.eeg_encoder.save_weights(result_path + "EEG_encoder_weights_" + str(fold) + ".hdf5")

    plt.figure()
    plt.plot(train_loss_history)
    plt.plot(val_loss_history)
    plt.title('Contrastive Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.savefig(result_path + "ContrastiveLoss.png")
    plt.show()
