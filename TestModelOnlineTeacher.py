import tensorflow as tf
from KnowledgeDistillation.Models.EnsembleDistillModel import EnsembleStudentOneDim
from KnowledgeDistillation.Models.EnsembleFeaturesModel import EnsembleSeparateModel, EnsembleModel
from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, TRAINING_RESULTS_PATH, ROAD_ECG, SPLIT_TIME, STRIDE, ECG_N, N_CLASS, FEATURES_N
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetch, DataFetchRoad
from KnowledgeDistillation.Utils.Metrics import PCC, CCC, SAGR, SoftF1
import datetime
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Libs.Utils import calcAccuracyRegression, calcAccuracyArValRegression


sns.set_style("whitegrid")
sns.set_color_codes("dark")

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
initial_learning_rate = 1e-3
EPOCHS = 200
PRE_EPOCHS = 100
BATCH_SIZE = 1
th = 0.5
ALL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
wait = 10

results = pd.DataFrame(index=[], columns=["Fold",
                                          "Loss",
                                          "SoftF1",
                                          "RMSE_ar", "PCC_ar", "CCC_ar", "SAGR_ar",
                                          "RMSE_val", "PCC_val", "CCC_val", "SAGR_val",
                                          "Ar_high", "Ar_low", "Ar_med",
                                          "Val_positive", "Val_negative", "Val_neutral"])

# setting
# fold = str(sys.argv[1])
fold=1
for fold in range(1, 6):
    prev_val_loss = 1000
    wait_i = 0
    result_path = TRAINING_RESULTS_PATH + "Binary_ECG\\fold_" + str(fold) + "\\"
    # checkpoint_prefix = result_path + "model_student_ECG_KD_high"
    checkpoint_prefix2 = result_path + "model_teacher"
    # datagenerator
    testing_data = DATASET_PATH + "\\stride=0.2\\test_data_" + str(fold) + ".csv"
    validation_data = DATASET_PATH + "\\stride=0.2\\validation_data_" + str(fold) + ".csv"
    data_fetch = DataFetch(test_file=testing_data, validation_file=validation_data,
                           ECG_N=ECG_RAW_N, KD=False, training=False, teacher=True, ECG=True, high_only=False)
    generator = data_fetch.fetch

    test_generator = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=2),
        output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([FEATURES_N]), tf.TensorShape([N_CLASS]), (), ()))

    test_data = test_generator.batch(data_fetch.test_n)

    with strategy.scope():
        # load pretrained model
        # encoder model

        # model = EnsembleStudentOneDim(num_output=num_output)
        model = EnsembleSeparateModel(num_output=num_output, features_length=FEATURES_N).loadBaseModel(
            checkpoint_prefix=checkpoint_prefix2)

        # test
        loss_test = tf.keras.metrics.Mean()
        softf1_test = SoftF1()
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

    predictions = []
    print("Start testing: Fold", fold)
    with strategy.scope():

        def test_step(inputs, GLOBAL_BATCH_SIZE=0):
            # X = tf.expand_dims(inputs[4], -1)
            X = inputs[0]
            y_emotion = inputs[1]
            y_r_ar = tf.expand_dims(inputs[2], -1)
            y_r_val = tf.expand_dims(inputs[3], -1)
            z_em, z_r_ar, z_r_val, _, _ = model(X, training=False)
            classific_loss = model.classificationLoss(z_em, y_emotion, global_batch_size=GLOBAL_BATCH_SIZE)
            mse_loss, regress_loss = model.regressionLoss(z_r_ar, z_r_val, y_r_ar, y_r_val, shake_params=shake_params,
                                                          global_batch_size=GLOBAL_BATCH_SIZE)

            update_test_metrics(mse_loss + classific_loss, z=[tf.nn.sigmoid(z_em), z_r_ar, z_r_val],
                                y=[y_emotion, y_r_ar, y_r_val])
            # print(X)

            return z_r_ar, z_r_val, y_r_ar, y_r_val

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

    with strategy.scope():
        # `experimental_run_v2` replicates the provided computation and runs it
        # with the distributed input.

        @tf.function
        def distributed_test_step(dataset_inputs, GLOBAL_BATCH_SIZE):
            per_replica_losses = strategy.run(test_step, args=(dataset_inputs, GLOBAL_BATCH_SIZE))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


        it = 0
        ar_pred_label = []
        val_pred_label = []
        template = ("{}, {}, {}, {}")
        shake_params = tf.random.uniform(shape=(3,), minval=0.1, maxval=1)
        for step, test in enumerate(test_data):
            prediction_ar, prediction_val, label_r_ar, label_r_val = distributed_test_step(test, data_fetch.test_n)
            # print(template.format(prediction_ar.numpy()[0, 0],  prediction_val.numpy()[0, 0], y_r_ar.numpy()[0, 0], y_r_val.numpy()[0, 0]))
            ar_pred_label.append([prediction_ar.numpy()[:, 0], label_r_ar.numpy()[:, 0]])
            val_pred_label.append([prediction_val.numpy()[:, 0], label_r_val.numpy()[:, 0]])

        ar_pred_label = np.concatenate(ar_pred_label).T
        val_pred_label = np.concatenate(val_pred_label).T

        # calcAccuracyRegression(ar_pred_label[:, 0], val_pred_label[:, 0], ar_pred_label[:, 1], val_pred_label[:, 1], mode="hard")
        # calcAccuracyRegression(ar_pred_label[:, 0], val_pred_label[:, 0], ar_pred_label[:, 1], val_pred_label[:, 1], mode="soft")
        # calcAccuracyRegression(ar_pred_label[:, 0], val_pred_label[:, 0], ar_pred_label[:, 1], val_pred_label[:, 1], mode="false")
        accuracy_ar, accuracy_val = calcAccuracyArValRegression(ar_pred_label[:, 0], val_pred_label[:, 0], ar_pred_label[:, 1], val_pred_label[:, 1])

        results_series = pd.Series([fold,
                                    loss_test.result().numpy(),
                                    softf1_test.result().numpy(),
                                    rmse_ar_test.result().numpy(),
                                    pcc_ar_test.result().numpy(),
                                    ccc_ar_test.result().numpy(),
                                    sagr_ar_test.result().numpy(),
                                    rmse_val_test.result().numpy(),
                                    pcc_val_test.result().numpy(),
                                    ccc_val_test.result().numpy(),
                                    sagr_val_test.result().numpy(),
                                    accuracy_ar[0], accuracy_ar[1], accuracy_ar[2],
                                    accuracy_val[0], accuracy_val[1], accuracy_val[2]],
                                   index=results.columns)
        results = results.append(results_series, ignore_index=True)

        # val positif
        a_p = (ar_pred_label[:, 0] > 0)
        a_n = (ar_pred_label[:, 0] < 0)
        a_p_results = np.sum(ar_pred_label[:, 1][a_p] > 0)
        a_n_results = np.sum(ar_pred_label[:, 1][a_n] <= 0)
        print((a_p_results + a_n_results) / (np.sum(a_p) + np.sum(a_n)))
        # val positif
        v_p = (val_pred_label[:, 0] > 0)
        v_n = (val_pred_label[:, 0] < 0)
        v_p_results = np.sum(val_pred_label[:, 1][v_p] > 0)
        v_n_results = np.sum(val_pred_label[:, 1][v_n] <= 0)
        print((v_p_results + v_n_results) / (np.sum(v_p) + np.sum(v_n)))

results.to_csv(TRAINING_RESULTS_PATH + "Binary_ECG\\AllResultsSummaryTeacher.csv", index=False)

