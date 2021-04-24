import tensorflow as tf
from KnowledgeDistillation.Models.EnsembleFeaturesModel import EnsembleModel, EnsembleSeparateModel
from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, TRAINING_RESULTS_PATH, N_CLASS, ECG_N
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetch
from Libs.Utils import regressLabelsConv, classifLabelsConv
import datetime
import os
import sys
from KnowledgeDistillation.Utils.Metrics import PCC, CCC, SAGR, SoftF1
import tensorflow_addons as tfa
import numpy as np

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
num_output = N_CLASS
initial_learning_rate = 1e-4
EPOCHS = 2000
PRE_EPOCHS = 100
BATCH_SIZE = 512
th = 0.5
ALL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
wait = 55
alpha = 0.35

# setting
# fold = str(sys.argv[1])
fold=1
prev_val_loss = 2000
wait_i = 0
result_path = TRAINING_RESULTS_PATH + "Binary_ECG\\regression+class(-3)\\fold_" + str(fold) + "\\"
checkpoint_prefix = result_path + "model_student_ECG_KD"

# tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = result_path + "tensorboard_student_ECG_KD\\" + current_time + '/train'
test_log_dir = result_path + "tensorboard_student_ECG_KD\\" + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# datagenerator

training_data = DATASET_PATH + "\\stride=0.2\\preliminary-results-data\\training_data_" + str(fold) + ".csv"
validation_data = DATASET_PATH + "\\stride=0.2\\preliminary-results-data\\validation_data_" + str(fold) + ".csv"
testing_data = DATASET_PATH + "\\stride=0.2\\preliminary-results-data\\test_data_" + str(fold) + ".csv"

data_fetch = DataFetch(train_file=training_data, test_file=testing_data, validation_file=validation_data,
                       ECG_N=ECG_RAW_N, KD=True, teacher=False, ECG=True, high_only=False)
generator = data_fetch.fetch

train_generator = tf.data.Dataset.from_generator(
    lambda: generator(training_mode=0),
    output_types=(tf.float32, tf.float32,  tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([FEATURES_N]), (tf.TensorShape([N_CLASS])), (), (), tf.TensorShape([ECG_N]), ()))

val_generator = tf.data.Dataset.from_generator(
    lambda: generator(training_mode=1),
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([FEATURES_N]), (tf.TensorShape([N_CLASS])), (), (), tf.TensorShape([ECG_N]), ()))

test_generator = tf.data.Dataset.from_generator(
    lambda: generator(training_mode=2),
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([FEATURES_N]), (tf.TensorShape([N_CLASS])), (), (), tf.TensorShape([ECG_N]), ()))

# train dataset
train_data = train_generator.shuffle(data_fetch.train_n, reshuffle_each_iteration=True).batch(ALL_BATCH_SIZE)

val_data = val_generator.batch(ALL_BATCH_SIZE)

test_data = test_generator.batch(ALL_BATCH_SIZE)

with strategy.scope():
    # model = EnsembleStudent(num_output=num_output, expected_size=EXPECTED_ECG_SIZE)

    # load pretrained model
    checkpoint_prefix_base = result_path + "model_teacher"
    teacher_model = EnsembleSeparateModel(num_output=num_output, features_length=FEATURES_N).loadBaseModel(
        checkpoint_prefix_base)
    # encoder model
    checkpoint_prefix_encoder = result_path + "model_base_student"

    model = EnsembleModel(num_output=num_output)
    total_steps = int((data_fetch.train_n / BATCH_SIZE) * EPOCHS)
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                   decay_steps=(EPOCHS / 2), decay_rate=0.95,
                                                                   staircase=True)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = tfa.optimizers.RectifiedAdam(learning_rate=initial_learning_rate, total_steps=total_steps, warmup_proportion=0.3, min_lr=1e-5)
    # ---------------------------Epoch&Loss--------------------------#
    # metrics
    # train
    loss_train = tf.keras.metrics.Mean()
    #soft f1
    softf1_train = SoftF1()
    softf1_test = SoftF1()
    # arousal
    rmse_ar_train = tf.keras.metrics.RootMeanSquaredError()
    pcc_ar_train = PCC()
    ccc_ar_train = CCC()
    sagr_ar_train = SAGR()
    # valence
    rmse_val_train = tf.keras.metrics.RootMeanSquaredError()
    pcc_val_train = PCC()
    ccc_val_train = CCC()
    sagr_val_train = SAGR()

    # test
    loss_test = tf.keras.metrics.Mean()
    # arousal
    rmse_ar_test = tf.keras.metrics.RootMeanSquaredError()
    pcc_ar_test = PCC()
    ccc_ar_test = CCC()
    sagr_ar_test = SAGR()
    # valence
    rmse_val_test = tf.keras.metrics.RootMeanSquaredError()
    pcc_val_test = PCC()
    ccc_val_test = CCC()
    sagr_val_test = SAGR()

# Manager
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), student_model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
# checkpoint.restore(manager.latest_checkpoint)

with strategy.scope():
    def train_step(inputs, shake_params, GLOBAL_BATCH_SIZE):
        X_t = inputs[0]
        X = inputs[4]
        w = inputs[5]
        # print(X)

        y_emotion = inputs[1]

        y_r_ar = tf.expand_dims(inputs[2], -1)
        y_r_val = tf.expand_dims(inputs[3], -1)

        with tf.GradientTape() as tape:
            t_em, t_r_ar, t_r_val, _, t_z = teacher_model(X_t, False)
            # t_em = tf.nn.sigmoid(t_em)
            # using latent
            # _, latent = base_model(X)
            z_em, z_r_ar, z_r_val, z = model(X, training=True)
            classific_loss = teacher_model.classificationLoss(z_em, y_emotion, global_batch_size=GLOBAL_BATCH_SIZE) # classification student-gt
            classific_distill_loss = teacher_model.classificationLoss(z_em, t_em, global_batch_size=GLOBAL_BATCH_SIZE) # classification student-teacher
            mse_loss, regress_loss = teacher_model.regressionLoss(z_r_ar, z_r_val, y_r_ar, y_r_val, shake_params=shake_params,
                                                global_batch_size=GLOBAL_BATCH_SIZE, sample_weight=w)# regression student-gt
            _, regress_distill_loss, mask = teacher_model.regressionDistillLoss(z_r_ar, z_r_val, y_r_ar, y_r_val,
                                                                                t_r_ar, t_r_val,
                                                                                shake_params=shake_params,
                                                                                global_batch_size=GLOBAL_BATCH_SIZE)  # regression student-teacher
            latent_loss = teacher_model.latentLoss(z, t_z, global_batch_size=GLOBAL_BATCH_SIZE, sample_weight=mask)


            # print(t_x)
            # print(z_x)


            classification_final_loss = classific_loss + alpha * classific_distill_loss
            regression_final_loss = regress_loss + alpha * regress_distill_loss
            # classification_final_loss = classific_loss
            # regression_final_loss = regress_loss
            final_loss = classification_final_loss + regression_final_loss + latent_loss

        # update gradient
        grads = tape.gradient(final_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        update_train_metrics(mse_loss, z=[tf.nn.sigmoid(z_em), z_r_ar, z_r_val], y=[y_emotion, y_r_ar, y_r_val])

        return final_loss


    def test_step(inputs, GLOBAL_BATCH_SIZE):
        X = inputs[4]
        w = inputs[5]

        y_emotion = inputs[1]

        y_r_ar = tf.expand_dims(inputs[2], -1)
        y_r_val = tf.expand_dims(inputs[3], -1)

        z_em, z_r_ar, z_r_val, _ = model(X, training=False)
        classific_loss = teacher_model.classificationLoss(z_em, y_emotion,
                                                  global_batch_size=GLOBAL_BATCH_SIZE)  # classification student-gt
        mse_loss, regress_loss = teacher_model.regressionLoss(z_r_ar, z_r_val, y_r_ar, y_r_val, shake_params=shake_params,
                                                      global_batch_size=GLOBAL_BATCH_SIZE, sample_weight=w)  # regression student-gt

        final_loss = regress_loss

        update_test_metrics( mse_loss, z=[tf.nn.sigmoid(z_em), z_r_ar, z_r_val], y=[y_emotion, y_r_ar, y_r_val])

        return final_loss


    def reset_metrics():
        # train
        loss_train.reset_states()
        # arousal
        rmse_ar_train.reset_states()
        pcc_ar_train.reset_states()
        ccc_ar_train.reset_states()
        sagr_ar_train.reset_states()
        # valence
        rmse_val_train.reset_states()
        pcc_val_train.reset_states()
        ccc_val_train.reset_states()
        sagr_val_train.reset_states()

        # test
        loss_test.reset_states()
        # arousal
        rmse_ar_test.reset_states()
        pcc_ar_test.reset_states()
        ccc_ar_test.reset_states()
        sagr_ar_test.reset_states()
        # valence
        rmse_val_test.reset_states()
        pcc_val_test.reset_states()
        ccc_val_test.reset_states()
        sagr_val_test.reset_states()

        #soft f1
        softf1_train.reset_states()
        softf1_test.reset_states()


    def update_train_metrics(loss, y, z):
        z_em, z_r_ar, z_r_val = z  # logits
        y_em, y_r_ar, y_r_val = y  # ground truth
        # train
        loss_train(loss)
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
        # soft f1
        softf1_train(y_em, z_em)


    def update_test_metrics(loss, y, z):
        z_em, z_r_ar, z_r_val = z  # logits
        y_em, y_r_ar, y_r_val = y  # ground truth
        # train
        loss_test(loss)
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
        # soft f1
        softf1_test(y_em, z_em)


    def write_train_tensorboard(epoch):
        tf.summary.scalar('Loss', loss_train.result(), step=epoch)
        #soft f1
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

        for step, val in enumerate(val_data):
            distributed_test_step(val, ALL_BATCH_SIZE)

        with train_summary_writer.as_default():
            write_train_tensorboard(epoch)
        with test_summary_writer.as_default():
            write_test_tensorboard(epoch)

        template = (
            "epoch {} | Train_loss: {} | Val_loss: {}")
        train_loss = loss_train.result().numpy()
        test_loss = loss_test.result().numpy()
        print(template.format(epoch + 1, train_loss, test_loss))

        # Save model

        if (prev_val_loss > test_loss):
            prev_val_loss = test_loss
            wait_i = 0
            manager.save()
        else:
            wait_i += 1
        if (wait_i == wait):
            break
        # reset state

        reset_metrics()

    checkpoint.restore(manager.latest_checkpoint)
    print("-------------------------------------------Testing----------------------------------------------")
    for step, test in enumerate(test_data):
        distributed_test_step(test, ALL_BATCH_SIZE)
    template = (
        "Test: loss: {}, rmse_ar: {}, ccc_ar: {}, pcc_ar: {}, sagr_ar: {} | rmse_val: {}, ccc_val: {},  pcc_val: {}, sagr_val: {}, softf1_val: {}")
    sys.stdout = open(result_path + "summary_student_ECG_KD.txt", "w")
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
