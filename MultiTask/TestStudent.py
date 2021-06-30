import tensorflow as tf
from MultiTask.MultiTaskModel import EnsembleSeparateModel, EnsembleModel
from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, \
    TRAINING_RESULTS_PATH, ROAD_ECG, SPLIT_TIME, STRIDE, ECG_N, PPG_N, EDA_N, Resp_N, N_CLASS, N_SUBJECT, N_VIDEO_GENRE
from MultiTask.DataGenerator import DataFetch
from Libs.Utils import calcAccuracyRegression, calcAccuracyArValRegression
from KnowledgeDistillation.Utils.Metrics import PCC, CCC, SAGR, SoftF1
import os
import numpy as np
import pandas as pd
import seaborn as sns

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
                                          "Val_positive", "Val_negative", "Val_neutral",
                                          "Acc_subject", "Acc_gender", "Acc_video_genre"])
for fold in range(1, 2):
    prev_val_loss = 1000
    wait_i = 0
    result_path = TRAINING_RESULTS_PATH + "MultiTask\\fold_" + str(fold) + "\\"
    # checkpoint_prefix = result_path + "model_student_ECG_KD_high"
    checkpoint_prefix2 = result_path + "model_student_ECG_KD"
    # datagenerator
    testing_data = DATASET_PATH + "\\stride=0.2_multitask\\test_data_" + str(fold) + ".csv"
    validation_data = DATASET_PATH + "\\stride=0.2_multitask\\validation_data_" + str(fold) + ".csv"
    data_fetch = DataFetch(test_file=testing_data, validation_file=validation_data,
                           ECG_N=ECG_RAW_N, KD=True, training=False, teacher=False,
                           ECG=True, high_only=False, multi_task=True)
    generator = data_fetch.fetch

    test_generator = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=2),
        # output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        # output_shapes=(tf.TensorShape([FEATURES_N]), (tf.TensorShape([N_CLASS])), (), (), tf.TensorShape([ECG_RAW_N]), ()))
        output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([FEATURES_N]), tf.TensorShape([N_CLASS]), (), (), tf.TensorShape([ECG_N]), (), tf.TensorShape([N_SUBJECT]), (), tf.TensorShape([N_VIDEO_GENRE])))

    test_data = test_generator.batch(data_fetch.test_n)

    with strategy.scope():
        # load pretrained model
        # encoder model

        # model = EnsembleStudentOneDim(num_output=num_output)
        checkpoint_prefix_base = result_path + "model_teacher"
        teacher_model = EnsembleSeparateModel(num_output=num_output, num_subject=N_SUBJECT, num_video_genre=N_VIDEO_GENRE, features_length=FEATURES_N).loadBaseModel(
            checkpoint_prefix_base)
        model = EnsembleModel(num_output=num_output, num_subject=N_SUBJECT, num_video_genre=N_VIDEO_GENRE).loadBaseModel(checkpoint_prefix=checkpoint_prefix2)

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
        # Sub-task accuracy
        subject_acc_test = tf.keras.metrics.CategoricalAccuracy()
        gender_acc_test = tf.keras.metrics.BinaryAccuracy()
        video_acc_test = tf.keras.metrics.CategoricalAccuracy()

    predictions = []
    print("Start testing: Fold", fold)
    with strategy.scope():
        def test_step(inputs, GLOBAL_BATCH_SIZE=0):
            # X = tf.expand_dims(inputs[4], -1)
            X = inputs[4]
            w = inputs[5]
            y_emotion = inputs[1]

            y_r_ar = tf.expand_dims(inputs[2], -1)
            y_r_val = tf.expand_dims(inputs[3], -1)
            y_sub = inputs[6]
            y_gen = tf.expand_dims(inputs[7], -1)
            y_video = inputs[8]
            z_em, z_r_ar, z_r_val, _, z_sub, z_gen, z_video = model(X, training=False)

            mse_loss, regress_loss = teacher_model.regressionLoss(z_r_ar, z_r_val, y_r_ar, y_r_val,
                                                                  shake_params=shake_params,
                                                                  global_batch_size=GLOBAL_BATCH_SIZE,
                                                                  sample_weight=w)  # regression student-gt

            update_test_metrics(mse_loss,
                                z=[tf.nn.sigmoid(z_em), z_r_ar, z_r_val, tf.nn.softmax(z_sub), tf.nn.sigmoid(z_gen), tf.nn.softmax(z_video)],
                                y=[y_emotion, y_r_ar, y_r_val, y_sub, y_gen, y_video])

            return z_r_ar, z_r_val, y_r_ar, y_r_val

        def update_test_metrics(loss, y, z):
            z_em, z_r_ar, z_r_val, z_sub, z_gen, z_video = z  # logits
            y_em, y_r_ar, y_r_val, y_sub, y_gen, y_video = y  # ground truth
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
            # Sub-task
            subject_acc_test(y_sub, z_sub)
            gender_acc_test(y_gen, z_gen)
            video_acc_test(y_video, z_video)

    with strategy.scope():
        # `experimental_run_v2` replicates the provided computation and runs it
        # with the distributed input.

        @tf.function
        def distributed_test_step(dataset_inputs, GLOBAL_BATCH_SIZE):
            per_replica_losses = strategy.run(test_step,
                                              args=(dataset_inputs, GLOBAL_BATCH_SIZE))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)


        it = 0
        ar_pred_label = []
        val_pred_label = []
        template = ("{}, {}, {}, {}")
        shake_params = tf.random.uniform(shape=(3,), minval=0.1, maxval=1)
        for step, test in enumerate(test_data):
            prediction_ar, prediction_val, y_r_ar, y_r_val = distributed_test_step(test, data_fetch.test_n)
            # print(template.format(prediction_ar.numpy()[0, 0],  prediction_val.numpy()[0, 0], y_r_ar.numpy()[0, 0], y_r_val.numpy()[0, 0]))
            ar_pred_label.append([prediction_ar.numpy()[:, 0], y_r_ar.numpy()[:, 0]])
            val_pred_label.append([prediction_val.numpy()[:, 0], y_r_val.numpy()[:, 0]])

        ar_pred_label = np.concatenate(ar_pred_label).T
        val_pred_label = np.concatenate(val_pred_label).T

        # sys.stdout = open(result_path + "accuracy_student_ECG_KD.txt", "w")
        # calcAccuracyRegression(ar_results[:, 0], val_results[:, 0], ar_results[:, 1], val_results[:, 1], mode="hard",
        #                        th=th)
        # calcAccuracyRegression(ar_results[:, 0], val_results[:, 0], ar_results[:, 1], val_results[:, 1], mode="soft",
        #                        th=th)
        # calcAccuracyRegression(ar_results[:, 0], val_results[:, 0], ar_results[:, 1], val_results[:, 1], mode="false",
        #                        th=th)
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
                                    accuracy_val[0], accuracy_val[1], accuracy_val[2],
                                    subject_acc_test.result().numpy(),
                                    gender_acc_test.result().numpy(),
                                    video_acc_test.result().numpy()],
                                   index=results.columns)
        results = results.append(results_series, ignore_index=True)

results.to_csv(TRAINING_RESULTS_PATH + "MultiTask\\AllResultsSummaryStudent_ECG_3tasks.csv", index=False)
