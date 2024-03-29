import tensorflow as tf
from KnowledgeDistillation.Models.EnsembleFeaturesModel import EnsembleSeparateModel
from Conf.Settings import FEATURES_N, DATASET_PATH, ECG_RAW_N, CHECK_POINT_PATH, TENSORBOARD_PATH, TRAINING_RESULTS_PATH, N_CLASS
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetch
from KnowledgeDistillation.Utils.Metrics import PCC, SAGR, CCC, SoftF1
import datetime
import os
import numpy as np
import sys
import tensorflow_addons as tfa

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
num_output = N_CLASS
initial_learning_rate = 1e-3
EPOCHS = 3000
BATCH_SIZE = 256
th = 0.5
ALL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
wait = 15


# setting
fold = str(sys.argv[1])
# fold=1
#setting model
prev_val_loss = 1000
wait_i = 0
result_path = TRAINING_RESULTS_PATH + "Binary_ECG\\regression+class(-3)\\fold_" + str(fold) + "\\"
checkpoint_prefix = result_path + "model_teacher"
# tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = result_path + "tensorboard_teacher\\" + current_time + '/train'
test_log_dir = result_path + "tensorboard_teacher\\" + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# datagenerator

training_data = DATASET_PATH + "\\stride=0.2\\preliminary-results-data\\training_data_" + str(fold) + ".csv"
validation_data = DATASET_PATH + "\\stride=0.2\\preliminary-results-data\\validation_data_" + str(fold) + ".csv"
testing_data = DATASET_PATH + "\\stride=0.2\\preliminary-results-data\\test_data_" + str(fold) + ".csv"

data_fetch = DataFetch(train_file=training_data, test_file=testing_data, validation_file=validation_data,
                       ECG_N=ECG_RAW_N, teacher=True,
                       KD=False, high_only=False)
generator = data_fetch.fetch

train_generator = tf.data.Dataset.from_generator(
    lambda: generator(),
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32 ),
    output_shapes=(tf.TensorShape([FEATURES_N]), tf.TensorShape([N_CLASS]), (), ()))

val_generator = tf.data.Dataset.from_generator(
    lambda: generator(training_mode=1),
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([FEATURES_N]), tf.TensorShape([N_CLASS]), (), ()))

test_generator = tf.data.Dataset.from_generator(
    lambda: generator(training_mode=2),
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([FEATURES_N]), tf.TensorShape([N_CLASS]), (), ()))

# train dataset
train_data = train_generator.shuffle(data_fetch.train_n).batch(ALL_BATCH_SIZE)

val_data = val_generator.batch(ALL_BATCH_SIZE)

test_data = test_generator.batch(ALL_BATCH_SIZE)

with strategy.scope():
    model = EnsembleSeparateModel(num_output=num_output, features_length=FEATURES_N)
    total_steps = int((data_fetch.train_n / ALL_BATCH_SIZE) * EPOCHS)
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                   decay_steps=EPOCHS, decay_rate=0.95, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = tfa.optimizers.RectifiedAdam(learning_rate=initial_learning_rate, total_steps=total_steps,
    #                                          warmup_proportion=0.1, min_lr=1e-5)

    # ---------------------------Loss & Metrics--------------------------#
    # loss
    loss_train = tf.keras.metrics.Mean()
    loss_test = tf.keras.metrics.Mean()

    #soft f1
    softf1_train = SoftF1()
    softf1_test = SoftF1()
    #rmse
    #train
    rmse_ar_train = tf.keras.metrics.RootMeanSquaredError()
    rmse_val_train = tf.keras.metrics.RootMeanSquaredError()
    #val
    rmse_ar_test = tf.keras.metrics.RootMeanSquaredError()
    rmse_val_test = tf.keras.metrics.RootMeanSquaredError()

    #pcc
    #train
    pcc_ar_train = PCC()
    pcc_val_train = PCC()
    #test
    pcc_ar_test = PCC()
    pcc_val_test = PCC()
    #ccc
    #train
    ccc_ar_train = CCC()
    ccc_val_train = CCC()
    #test
    ccc_ar_test = CCC()
    ccc_val_test = CCC()
    #sagr
    #train
    sagr_ar_train = SAGR()
    sagr_val_train = SAGR()
    #test
    sagr_ar_test = SAGR()
    sagr_val_test = SAGR()


    # Manager
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), teacher_model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
    # checkpoint.restore(manager.latest_checkpoint)

with strategy.scope():
    def train_step(inputs, shake_params, GLOBAL_BATCH_SIZE=0):
        X = inputs[0]
        # print(X)
        y_emotion = inputs[1]

        y_r_ar = tf.expand_dims(inputs[2], -1)
        y_r_val = tf.expand_dims(inputs[3], -1)

        with tf.GradientTape() as tape_ar:
            z_em, z_r_ar, z_r_val, rec_X, _ = model(X, training=True)
            classific_loss = model.classificationLoss(z_em, y_emotion, global_batch_size=GLOBAL_BATCH_SIZE)
            mse_loss, regress_loss = model.regressionLoss(z_r_ar, z_r_val, y_r_ar, y_r_val, shake_params=shake_params,
                                                global_batch_size=GLOBAL_BATCH_SIZE)
            rec_loss = model.reconstructLoss(X, rec_X, global_batch_size=GLOBAL_BATCH_SIZE)

            final_loss = regress_loss + classific_loss + rec_loss

        # update gradient
        grads = tape_ar.gradient(final_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        update_train_metrics(mse_loss + classific_loss, z=[tf.nn.sigmoid(z_em), z_r_ar, z_r_val], y=[y_emotion, y_r_ar, y_r_val])

        return final_loss


    def test_step(inputs, GLOBAL_BATCH_SIZE=0):
        X = inputs[0]
        y_emotion = inputs[1]

        y_r_ar = tf.expand_dims(inputs[2], -1)
        y_r_val = tf.expand_dims(inputs[3], -1)
        z_em, z_r_ar, z_r_val, _, _ = model(X, training=False)
        classific_loss = model.classificationLoss(z_em, y_emotion, global_batch_size=GLOBAL_BATCH_SIZE)
        mse_loss, regress_loss = model.regressionLoss(z_r_ar, z_r_val, y_r_ar, y_r_val, shake_params=shake_params,
                                                     global_batch_size=GLOBAL_BATCH_SIZE)

        final_loss = regress_loss + classific_loss

        update_test_metrics(mse_loss + classific_loss, z=[tf.nn.sigmoid(z_em), z_r_ar, z_r_val], y=[y_emotion, y_r_ar, y_r_val])

        return final_loss


    def reset_states():
        # loss
        loss_train.reset_states()
        loss_test.reset_states()
        # soft f1
        softf1_train.reset_states()
        softf1_test.reset_states()
        # rmse
        # train
        rmse_ar_train.reset_states()
        rmse_val_train.reset_states()
        # val
        rmse_ar_test.reset_states()
        rmse_val_test.reset_states()
        # pcc
        # train
        pcc_ar_train.reset_states()
        pcc_val_train.reset_states()
        # test
        pcc_ar_test.reset_states()
        pcc_val_test.reset_states()
        # ccc
        # train
        ccc_ar_train.reset_states()
        ccc_val_train.reset_states()
        # test
        ccc_ar_test.reset_states()
        ccc_val_test.reset_states()
        # sagr
        # train
        sagr_ar_train.reset_states()
        sagr_val_train.reset_states()
        # test
        sagr_ar_test.reset_states()
        sagr_val_test.reset_states()
        #soft f1
        softf1_train.reset_states()
        softf1_test.reset_states()


    def update_train_metrics(loss, y, z):
        z_em, z_r_ar, z_r_val = z  # logits
        y_em, y_r_ar, y_r_val = y  # ground truth
        # loss
        loss_train(loss)
        # soft f1
        softf1_train(y_em, z_em)
        # arousal
        rmse_ar_train(y_r_ar, z_r_ar)
        pcc_ar_train(y_r_ar, z_r_ar)
        ccc_ar_train(y_r_ar, z_r_ar)
        sagr_ar_train(y_r_ar, z_r_ar)
        # valence
        rmse_val_train(y_r_val, z_r_val)
        pcc_val_train(y_r_val, z_r_val)
        ccc_val_train(y_r_val, z_r_val)
        sagr_val_train(y_r_val, z_r_val)

    def update_test_metrics(loss, y, z):
        z_em, z_r_ar, z_r_val = z  # logit
        y_em, y_r_ar, y_r_val = y  # ground truth
        # loss
        loss_test(loss)
        # soft f1
        softf1_test(y_em, z_em)
        # arousal
        rmse_ar_test(y_r_ar, z_r_ar)
        pcc_ar_test(y_r_ar, z_r_ar)
        ccc_ar_test(y_r_ar, z_r_ar)

        sagr_ar_test(y_r_ar, z_r_ar)
        # valence
        rmse_val_test(y_r_val, z_r_val)
        pcc_val_test(y_r_val, z_r_val)
        ccc_val_test(y_r_val, z_r_val)
        sagr_val_test(y_r_val, z_r_val)



    def write_train_tensorboard(epoch):
        tf.summary.scalar('Loss', loss_train.result(), step=epoch)
        # soft f1
        tf.summary.scalar('Soft F1', np.mean(softf1_train.result()), step=epoch)
        # arousal
        tf.summary.scalar('RMSE arousal', rmse_ar_train.result(), step=epoch)
        tf.summary.scalar('PCC arousal', pcc_ar_train.result(), step=epoch)
        tf.summary.scalar('CCC arousal', ccc_ar_train.result(), step=epoch)
        tf.summary.scalar('SAGR arousal', sagr_ar_train.result(), step=epoch)
        # valence
        tf.summary.scalar('RMSE valence', rmse_val_train.result(), step=epoch)
        tf.summary.scalar('PCC valence', pcc_val_train.result(), step=epoch)
        tf.summary.scalar('CCC valence', ccc_val_train.result(), step=epoch)
        tf.summary.scalar('SAGR valence', sagr_val_train.result(), step=epoch)

    def write_test_tensorboard(epoch):
        tf.summary.scalar('Loss', loss_test.result(), step=epoch)
        # soft f1
        tf.summary.scalar('Soft F1', np.mean(softf1_test.result()), step=epoch)
        # arousal
        tf.summary.scalar('RMSE arousal', rmse_ar_test.result(), step=epoch)
        tf.summary.scalar('PCC arousal', pcc_ar_test.result(), step=epoch)
        tf.summary.scalar('CCC arousal', ccc_ar_test.result(), step=epoch)
        tf.summary.scalar('SAGR arousal', sagr_ar_test.result(), step=epoch)
        # valence
        tf.summary.scalar('RMSE valence', rmse_val_test.result(), step=epoch)
        tf.summary.scalar('PCC valence', pcc_val_test.result(), step=epoch)
        tf.summary.scalar('CCC valence', ccc_val_test.result(), step=epoch)
        tf.summary.scalar('SAGR valence', sagr_val_test.result(), step=epoch)



with strategy.scope():
    # `experimental_run_v2` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs, shake_params, GLOBAL_BATCH_SIZE):
        per_replica_losses = strategy.run(train_step,
                                          args=(dataset_inputs, shake_params, GLOBAL_BATCH_SIZE))
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
        shake_params = tf.random.uniform(shape=(3,), minval=0.1, maxval=1)
        for step, train in enumerate(train_data):
            # print(tf.reduce_max(train[0][0]))
            distributed_train_step(train, shake_params, ALL_BATCH_SIZE)
            it += 1

        with train_summary_writer.as_default():
            write_train_tensorboard(epoch)

        for step, val in enumerate(val_data):
            distributed_test_step(val, ALL_BATCH_SIZE)

        with test_summary_writer.as_default():
            write_test_tensorboard(epoch)

        template = (
            "epoch {} | Train_loss: {:.4f} | Val_loss: {}")
        print(template.format(epoch + 1, loss_train.result().numpy(), loss_test.result().numpy()))

        if (prev_val_loss > loss_test.result().numpy()):
            prev_val_loss = loss_test.result().numpy()
            wait_i = 0
            manager.save()
        else:
            wait_i += 1
        if (wait_i == wait):
            break


    print("-------------------------------------------Testing----------------------------------------------")
    for step, test in enumerate(test_data):
        distributed_test_step(test, ALL_BATCH_SIZE)
    template = (
        "Test: loss: {}, rmse_ar: {}, ccc_ar: {}, pcc_ar: {}, sagr_ar: {} | rmse_val: {}, ccc_val: {},  pcc_val: {}, sagr_val: {}, softf1_val: {}")
    sys.stdout = open(result_path + "summary_teacher.txt", "w")
    print(template.format(
        loss_test.result().numpy(),
        rmse_ar_test.result().numpy(),
        ccc_ar_test.result().numpy(),
        pcc_ar_test.result().numpy(),
        sagr_ar_test.result().numpy(),
        rmse_val_test.result().numpy(),
        ccc_val_train.result().numpy(),
        pcc_val_test.result().numpy(),
        sagr_val_test.result().numpy(),
        np.mean(softf1_test.result().numpy())
    ))
    sys.stdout.close()

