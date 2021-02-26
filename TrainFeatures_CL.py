import tensorflow as tf
import matplotlib.pyplot as plt
from ContrastiveLearning.Models.ContrastiveLearningModels import ECGEEGEncoder
from ContrastiveLearning.Models.ContrastiveLearningModels import ClassifyArVal_CL
from Conf.Settings import DATASET_PATH, CHECK_POINT_PATH, ECG_RAW_N, EEG_N
from ContrastiveLearning.DataGenerator_CL import DataFetchPreTrain_CL
from tensorflow.python.keras.utils.vis_utils import plot_model
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
NUM_OUTPUT = 2
EPOCHS = 200
BATCH_SIZE = 512
FINE_TUNING = False
ALL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
WAIT = 10
initial_learning_rate = 1e-3
print("All batch size:", ALL_BATCH_SIZE)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
for fold in range(1, 6):
    prev_vald_loss_ar = 1000
    prev_vald_loss_val = 1000
    result_path = CHECK_POINT_PATH + "ClassifyArVal_CL\\" + current_time + "\\fold_" + str(fold) + "\\"
    checkpoint_prefix = result_path + "checkpoint\\"

    # tensorboard
    train_log_dir = result_path + "tensorboard\\train\\"
    test_log_dir = result_path + "tensorboard\\test\\"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    training_data = DATASET_PATH + "training_data_" + str(fold) + ".csv"
    validation_data = DATASET_PATH + "validation_data_" + str(fold) + ".csv"
    testing_data = DATASET_PATH + "test_data_" + str(fold) + ".csv"

    data_fetch = DataFetchPreTrain_CL(training_data, validation_data, testing_data, ECG_RAW_N, pretrain=False, eeg_raw=False)
    # data_fetch = DataFetchPreTrain_CL(validation_data, testing_data, testing_data, ECG_RAW_N, pretrain=False, eeg_raw=False)
    generator = data_fetch.fetch

    train_generator = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=0),
        output_types=(tf.float32, tf.float32, tf.int32, tf.int32, tf.int32),
        output_shapes=(
            tf.TensorShape([ECG_RAW_N]), tf.TensorShape([EEG_N]), tf.TensorShape([2]),
            tf.TensorShape([2]),
            tf.TensorShape([4])))

    val_generator = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=1),
        output_types=(tf.float32, tf.float32, tf.int32, tf.int32, tf.int32),
        output_shapes=(
            tf.TensorShape([ECG_RAW_N]), tf.TensorShape([EEG_N]), tf.TensorShape([2]),
            tf.TensorShape([2]),
            tf.TensorShape([4])))

    test_generator = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=2),
        output_types=(tf.float32, tf.float32, tf.int32, tf.int32, tf.int32),
        output_shapes=(
            tf.TensorShape([ECG_RAW_N]), tf.TensorShape([EEG_N]), tf.TensorShape([2]),
            tf.TensorShape([2]),
            tf.TensorShape([4])))

    train_data = train_generator.shuffle(data_fetch.train_n).batch(ALL_BATCH_SIZE)
    val_data = val_generator.batch(ALL_BATCH_SIZE)
    test_data = test_generator.batch(ALL_BATCH_SIZE)

    with strategy.scope():
        # model = EnsembleStudent(num_output=num_output, expected_size=EXPECTED_ECG_SIZE)

        # load pretrained model
        # checkpoint_prefix_base = CHECK_POINT_PATH + "fold_M" + str(fold)

        # Define model
        CL = ECGEEGEncoder()
        encoder_weight_path = CHECK_POINT_PATH + "ContrastiveLearning\\20210202-171606\\"
        input_ecg = tf.keras.layers.Input(shape=(ECG_RAW_N,))
        input_eeg = tf.keras.layers.Input(shape=(EEG_N,))
        CL.createModel(input_ecg, input_eeg)
        CL.ecg_encoder.load_weights(encoder_weight_path + "ECG_encoder_weights_1.hdf5")
        classification = ClassifyArVal_CL(num_output=NUM_OUTPUT, ecg_encoder=CL.ecg_encoder,
                                          fine_tuning=FINE_TUNING)
        classification.createModel()
        classification.model.summary()
        plot_model(classification.model, to_file="ClassificationModel.png", show_shapes=True)
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                       decay_steps=EPOCHS, decay_rate=0.95,
                                                                       staircase=True)
        # learning_rate = initial_learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # ---------------------------Epoch&Loss--------------------------#
        # loss
        train_loss_ar = tf.keras.metrics.Mean()
        vald_loss_ar = tf.keras.metrics.Mean()
        train_loss_val = tf.keras.metrics.Mean()
        vald_loss_val = tf.keras.metrics.Mean()

        # accuracy
        train_acc_ar = tf.keras.metrics.BinaryAccuracy()
        vald_acc_ar = tf.keras.metrics.BinaryAccuracy()
        train_acc_val = tf.keras.metrics.BinaryAccuracy()
        vald_acc_val = tf.keras.metrics.BinaryAccuracy()

        # precision
        train_pre_ar = tf.keras.metrics.Precision()
        vald_pre_ar = tf.keras.metrics.Precision()
        train_pre_val = tf.keras.metrics.Precision()
        vald_pre_val = tf.keras.metrics.Precision()

        # recall
        train_rec_ar = tf.keras.metrics.Recall()
        vald_rec_ar = tf.keras.metrics.Recall()
        train_rec_val = tf.keras.metrics.Recall()
        vald_rec_val = tf.keras.metrics.Recall()

    # Manager
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, base_model=classification.model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=10)
    checkpoint.restore(manager.latest_checkpoint)

    with strategy.scope():
        def train_step(x, y_ar, y_val, global_batch_size):
            with tf.GradientTape() as tape:
                loss_ar, loss_val, pred_ar, pred_val = classification.computeLoss(x, y_ar, y_val, global_batch_size)
                final_loss = 0.5 * (loss_ar + loss_val)

            # update gradient
            grads = tape.gradient(final_loss, classification.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, classification.model.trainable_variables))

            # update_weights = CL.ecg_model.trainable_variables + CL.eeg_model.trainable_variables
            # grads = tape.gradient(final_loss, update_weights)
            # optimizer.apply_gradients(zip(grads, update_weights))

            train_loss_ar(loss_ar)
            train_loss_val(loss_val)
            train_acc_ar(y_ar, pred_ar)
            train_acc_val(y_val, pred_val)
            train_pre_ar(y_ar, pred_ar)
            train_pre_val(y_val, pred_val)
            train_rec_ar(y_ar, pred_ar)
            train_rec_val(y_val, pred_val)

            return final_loss


        def test_step(x, y_ar, y_val, global_batch_size):
            loss_ar, loss_val, pred_ar, pred_val = classification.computeLoss(x, y_ar, y_val, global_batch_size)
            final_loss = 0.5 * (loss_ar + loss_val)
            vald_loss_ar(loss_ar)
            vald_loss_val(loss_val)
            vald_acc_ar(y_ar, pred_ar)
            vald_acc_val(y_val, pred_val)
            vald_pre_ar(y_ar, pred_ar)
            vald_pre_val(y_val, pred_val)
            vald_rec_ar(y_ar, pred_ar)
            vald_rec_val(y_val, pred_val)

            return final_loss


        def train_reset_states():
            train_loss_ar.reset_states()
            train_loss_val.reset_states()
            train_acc_ar.reset_states()
            train_acc_val.reset_states()
            train_pre_ar.reset_states()
            train_pre_val.reset_states()
            train_rec_ar.reset_states()
            train_rec_val.reset_states()


        def vald_reset_states():
            vald_loss_ar.reset_states()
            vald_loss_val.reset_states()
            vald_acc_ar.reset_states()
            vald_acc_val.reset_states()
            vald_pre_ar.reset_states()
            vald_pre_val.reset_states()
            vald_rec_ar.reset_states()
            vald_rec_val.reset_states()

    with strategy.scope():
        # `experimental_run_v2` replicates the provided computation and runs it
        # with the distributed input.

        @tf.function
        def distributed_train_step(x, y_ar, y_val, global_batch_size):
            per_replica_losses = strategy.run(train_step,
                                              args=(x, y_ar, y_val, global_batch_size))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)


        @tf.function
        def distributed_test_step(x, y_ar, y_val, global_batch_size):
            per_replica_losses = strategy.run(test_step,
                                              args=(x, y_ar, y_val, global_batch_size))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)


        train_loss_history_ar = []
        vald_loss_history_ar = []
        train_acc_history_ar = []
        vald_acc_history_ar = []
        train_loss_history_val = []
        vald_loss_history_val = []
        train_acc_history_val = []
        vald_acc_history_val = []

        for epoch in range(EPOCHS):
            # Train Loop
            for train in train_data:
                x = train[0]
                y_ar = train[2]
                y_val = train[3]
                distributed_train_step(x, y_ar, y_val, global_batch_size=ALL_BATCH_SIZE)

            with train_summary_writer.as_default():
                tf.summary.scalar('Arousal Loss', train_loss_ar.result(), step=epoch)
                tf.summary.scalar('Valence Loss', train_loss_val.result(), step=epoch)
                tf.summary.scalar('Arousal Accuracy', train_acc_ar.result(), step=epoch)
                tf.summary.scalar('Valence Accuracy', train_acc_val.result(), step=epoch)
                tf.summary.scalar('Arousal Precision', train_pre_ar.result(), step=epoch)
                tf.summary.scalar('Valence Precision', train_pre_val.result(), step=epoch)
                tf.summary.scalar('Arousal Recall', train_rec_ar.result(), step=epoch)
                tf.summary.scalar('Valence Recall', train_rec_val.result(), step=epoch)

            # Validation Loop
            for val in val_data:
                x = val[0]
                y_ar = val[2]
                y_val = val[3]
                distributed_test_step(x, y_ar, y_val, global_batch_size=ALL_BATCH_SIZE)

            with test_summary_writer.as_default():
                tf.summary.scalar('Arousal Loss', vald_loss_ar.result(), step=epoch)
                tf.summary.scalar('Valence Loss', vald_loss_val.result(), step=epoch)
                tf.summary.scalar('Arousal Accuracy', vald_acc_ar.result(), step=epoch)
                tf.summary.scalar('Valence Accuracy', vald_acc_val.result(), step=epoch)
                tf.summary.scalar('Arousal Precision', vald_pre_ar.result(), step=epoch)
                tf.summary.scalar('Valence Precision', vald_pre_val.result(), step=epoch)
                tf.summary.scalar('Arousal Recall', vald_rec_ar.result(), step=epoch)
                tf.summary.scalar('Valence Recall', vald_rec_val.result(), step=epoch)

            train_loss_history_ar.append(train_loss_ar.result().numpy())
            vald_loss_history_ar.append(vald_loss_ar.result().numpy())
            train_acc_history_ar.append(train_acc_ar.result().numpy())
            vald_acc_history_ar.append(vald_acc_ar.result().numpy())
            train_loss_history_val.append(train_loss_val.result().numpy())
            vald_loss_history_val.append(vald_loss_val.result().numpy())
            train_acc_history_val.append(train_acc_val.result().numpy())
            vald_acc_history_val.append(vald_acc_val.result().numpy())

            print("Epoch: {}/{} | Train Loss | Ar: {:.4f}, Val: {:.4f} | Val Loss | Ar: {:.4f}, Val: {:.4f}".format(
                epoch + 1, EPOCHS, train_loss_ar.result().numpy(), train_loss_val.result().numpy(),
                vald_loss_ar.result().numpy(), vald_loss_val.result().numpy()))

            lr_now = optimizer._decayed_lr(tf.float32).numpy()
            print("Now learning rate:", lr_now)

            # Save model
            if prev_vald_loss_ar > vald_loss_ar.result().numpy() or prev_vald_loss_val > vald_loss_val.result().numpy():
                prev_vald_loss_ar = vald_loss_ar.result().numpy()
                prev_vald_loss_val = vald_loss_val.result().numpy()
                wait_i = 0
                manager.save()
            else:
                wait_i += 1
            if wait_i == WAIT:
                break

            # reset state
            train_reset_states()
            vald_reset_states()

    print("-------------------------------------------Testing----------------------------------------------")
    for test in test_data:
        x = test[0]
        y_ar = test[2]
        y_val = test[3]
        distributed_test_step(x, y_ar, y_val, global_batch_size=ALL_BATCH_SIZE)

    print("Arousal | Test | Loss: {}, Acc: {}, Precision: {}, Recall: {}".format(vald_loss_ar.result().numpy(),
                                                                                 vald_acc_ar.result().numpy(),
                                                                                 vald_pre_ar.result().numpy(),
                                                                                 vald_rec_ar.result().numpy()))
    print("Valence | Test | Loss: {}, Acc: {}, Precision: {}, Recall: {}".format(vald_loss_val.result().numpy(),
                                                                                 vald_acc_val.result().numpy(),
                                                                                 vald_pre_val.result().numpy(),
                                                                                 vald_rec_val.result().numpy()))

    vald_reset_states()
    print("-----------------------------------------------------------------------------------------")

    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(wspace=0.3)
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history_ar)
    plt.plot(vald_loss_history_ar)
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history_ar)
    plt.plot(vald_acc_history_ar)
    plt.title('Accuracy')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    # plt.tight_layout()
    plt.suptitle("Arousal Classification (CL)")
    plt.savefig(result_path + "ClassificationResult_Ar.png")

    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(wspace=0.3)
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history_val)
    plt.plot(vald_loss_history_val)
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history_val)
    plt.plot(vald_acc_history_val)
    plt.title('Accuracy')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    # plt.tight_layout()
    plt.suptitle("Valence Classification (CL)")
    plt.savefig(result_path + "ClassificationResult_Val.png")

    # plt.show()
