import tensorflow as tf
from KnowledgeDistillation.Models.EnsembleDistillModel import EnsembleStudentOneDim
from KnowledgeDistillation.Models.EnsembleFeaturesModel import EnsembleSeparateModel_MClass
from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, TRAINING_RESULTS_PATH
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetch
from Libs.Utils import regressLabelsConv, classifLabelsConv
import datetime
import os
import sys
from KnowledgeDistillation.Utils.Metrics import PCC, CCC, SAGR

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
num_output_ar = 3
num_output_val = 3
initial_learning_rate = 1.e-4
EPOCHS = 500
PRE_EPOCHS = 100
BATCH_SIZE = 32
th = 0.5
ALL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
wait = 10
alpha = 0.9

# setting
fold = str(sys.argv[1])
# fold=1
prev_val_loss = 1000
wait_i = 0
result_path = TRAINING_RESULTS_PATH + "Binary_ECG\\fold_" + str(fold) + "\\"
checkpoint_prefix = result_path + "model_student"

# tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = result_path + "tensorboard_student\\" + current_time + '/train'
test_log_dir = result_path + "tensorboard_student\\" + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# datagenerator

training_data = DATASET_PATH + "\\stride=0.2\\training_data_" + str(fold) + ".csv"
validation_data = DATASET_PATH + "\\stride=0.2\\validation_data_" + str(fold) + ".csv"
testing_data = DATASET_PATH + "\\stride=0.2\\test_data_" + str(fold) + ".csv"

data_fetch = DataFetch(train_file=training_data, test_file=testing_data, validation_file=validation_data,
                       ECG_N=ECG_RAW_N, KD=True, multiple=True)
generator = data_fetch.fetch

train_generator = tf.data.Dataset.from_generator(
    lambda: generator(training_mode=0),
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([FEATURES_N]), (), (), (), (),  tf.TensorShape([ECG_RAW_N])))

val_generator = tf.data.Dataset.from_generator(
    lambda: generator(training_mode=1),
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([FEATURES_N]), (), (), (), (), tf.TensorShape([ECG_RAW_N])))

test_generator = tf.data.Dataset.from_generator(
    lambda: generator(training_mode=2),
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([FEATURES_N]), (), (), (), (), tf.TensorShape([ECG_RAW_N])))

# train dataset
train_data = train_generator.shuffle(data_fetch.train_n).repeat(3).batch(ALL_BATCH_SIZE)

val_data = val_generator.batch(BATCH_SIZE)

test_data = test_generator.batch(BATCH_SIZE)

with strategy.scope():
    # model = EnsembleStudent(num_output=num_output, expected_size=EXPECTED_ECG_SIZE)

    # load pretrained model
    checkpoint_prefix_base = result_path + "model_teacher"
    teacher_model = EnsembleSeparateModel_MClass(num_output_val=3., num_output_ar=3).loadBaseModel(checkpoint_prefix_base)
    # encoder model
    checkpoint_prefix_encoder = result_path + "model_base_student"

    model = EnsembleStudentOneDim(num_output_ar=num_output_ar, num_output_val=num_output_val)

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                   decay_steps=(EPOCHS / 2), decay_rate=0.95,
                                                                   staircase=True)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # ---------------------------Epoch&Loss--------------------------#
    # metrics
    # train
    loss_train = tf.keras.metrics.Mean()
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
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, base_model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
# checkpoint.restore(manager.latest_checkpoint)

with strategy.scope():
    def train_step(inputs, shake_params, GLOBAL_BATCH_SIZE):
        # X = base_model.extractFeatures(inputs[-1])
        X_t = inputs[0]
        X = tf.expand_dims(inputs[-1], -1)
        # print(X)

        y_d_ar = tf.expand_dims(inputs[1], -1)
        y_d_val = tf.expand_dims(inputs[2], -1)

        y_r_ar = tf.expand_dims(inputs[3], -1)
        y_r_val = tf.expand_dims(inputs[4], -1)




        with tf.GradientTape() as tape:
            ar_logit, val_logit, z = teacher_model.predictKD(X_t)
            #using latent
            # _, latent = base_model(X)
            z_ar, z_val, z_r_ar, z_r_val = model(X, training=True)
            classific_loss = model.classificationLoss( z_ar, z_val, y_d_ar, y_d_val, ar_logit, val_logit, alpha, global_batch_size=GLOBAL_BATCH_SIZE)
            regress_loss = model.regressionLoss(z_r_ar, z_r_val, y_r_ar, y_r_val, shake_params=shake_params , global_batch_size=GLOBAL_BATCH_SIZE)


            final_loss = classific_loss + regress_loss

        # update gradient
        grads = tape.gradient(final_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        update_train_metrics(regress_loss, z = [z_r_ar, z_r_val], y =[ y_r_ar, y_r_val])


        return final_loss


    def test_step(inputs, GLOBAL_BATCH_SIZE):
        X = tf.expand_dims(inputs[-1], -1)

        y_r_ar = tf.expand_dims(inputs[3], -1)
        y_r_val = tf.expand_dims(inputs[4], -1)

        z_ar, z_val, z_r_ar, z_r_val = model(X, training=True)
        regress_loss = model.regressionLoss(z_r_ar, z_r_val, y_r_ar, y_r_val, training=False, global_batch_size=GLOBAL_BATCH_SIZE)

        final_loss = regress_loss



        update_test_metrics(regress_loss, z=[z_r_ar, z_r_val], y=[y_r_ar, y_r_val])

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


    def update_train_metrics(loss, y, z):
        z_r_ar, z_r_val = z  # logits
        y_r_ar, y_r_val = y  # ground truth
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


    def update_test_metrics(loss, y, z):
        z_r_ar, z_r_val = z  # logit
        y_r_ar, y_r_val = y  # ground truth
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


    def write_train_tensorboard(epoch):
        tf.summary.scalar('Loss', loss_train.result(), step=epoch)
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
        shake_params = tf.random.uniform(shape=(3, ), minval=0.1, maxval=1)
        for step, train in enumerate(train_data):
            # print(tf.reduce_max(train[0][0]))
            distributed_train_step(train, shake_params, ALL_BATCH_SIZE)
            it += 1


        for step, val in enumerate(val_data):
            distributed_test_step(val, data_fetch.val_n)

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


    print("-------------------------------------------Testing----------------------------------------------")
    checkpoint.restore(manager.latest_checkpoint)
    for step, test in enumerate(test_data):
        distributed_test_step(test, data_fetch.test_n)
    template = (
        "Test: loss: {}, rmse_ar: {}, ccc_ar: {}, pcc_ar: {}, sagr_ar: {} | rmse_val: {}, ccc_val: {},  pcc_val: {}, sagr_val: {}")

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
    ))


    reset_metrics()
    print("-----------------------------------------------------------------------------------------")
