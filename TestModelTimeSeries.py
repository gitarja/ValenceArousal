import tensorflow as tf
from KnowledgeDistillation.Models.EnsembleDistillModel import EnsembleStudentOneDim
from KnowledgeDistillation.Models.EnsembleFeaturesModel import EnsembleSeparateModel, EnsembleModel
from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, \
    TRAINING_RESULTS_PATH, ROAD_ECG, SPLIT_TIME, STRIDE, ECG_N, N_CLASS, FEATURES_N
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetch, DataFetchRoad
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

# setting
# fold = str(sys.argv[1])
fold = 1
prev_val_loss = 1000
wait_i = 0
result_path = TRAINING_RESULTS_PATH + "Binary_ECG\\fold_" + str(fold) + "\\"
checkpoint_prefix_teacher = result_path + "model_teacher"
checkpoint_prefix_student = result_path + "model_student_ECG_KD"
date = "2020-10-29"
subject = "E2"
# datagenerator
testing_data = DATASET_PATH + date + "\\" + subject + "-" + date + "\\features_list_0.2.csv"
# testing_data = DATASET_PATH + "\\stride=0.2\\all_data.csv"
validation_data = DATASET_PATH + "\\stride=0.2\\validation_data_" + str(fold) + ".csv"
data_fetch_teacher = DataFetch(test_file=testing_data, validation_file=validation_data,
                               ECG_N=ECG_RAW_N, KD=False, training=False, teacher=True, ECG=True, high_only=False)
data_fetch_student = DataFetch(test_file=testing_data, validation_file=validation_data,
                               ECG_N=ECG_RAW_N, KD=True, training=False, teacher=False, ECG=True, high_only=False)
generator_teacher = data_fetch_teacher.fetch
generator_student = data_fetch_student.fetch

test_generator_teacher = tf.data.Dataset.from_generator(
    lambda: generator_teacher(training_mode=2),
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([FEATURES_N]), tf.TensorShape([N_CLASS]), (), ()))
test_generator_student = tf.data.Dataset.from_generator(
    lambda: generator_student(training_mode=2),
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([FEATURES_N]), (tf.TensorShape([N_CLASS])), (), (), tf.TensorShape([ECG_N]), ()))

test_data_teacher = test_generator_teacher.batch(BATCH_SIZE)
test_data_student = test_generator_student.batch(BATCH_SIZE)

with strategy.scope():
    # load pretrained model
    # encoder model

    # model = EnsembleStudentOneDim(num_output=num_output)
    teacher_model = EnsembleSeparateModel(num_output=num_output, features_length=FEATURES_N).loadBaseModel(
        checkpoint_prefix=checkpoint_prefix_teacher)
    student_model = EnsembleModel(num_output=num_output).loadBaseModel(checkpoint_prefix=checkpoint_prefix_student)

predictions = []
print("Start testing")
with strategy.scope():
    def test_step_teacher(inputs, GLOBAL_BATCH_SIZE=0):
        # X = tf.expand_dims(inputs[4], -1)
        X = inputs[0]
        y_r_ar = tf.expand_dims(inputs[2], -1)
        y_r_val = tf.expand_dims(inputs[3], -1)
        print(X)
        _, prediction_ar, prediction_val, _, _ = teacher_model(X, training=False)

        return prediction_ar, prediction_val, y_r_ar, y_r_val


    def test_step_student(inputs, GLOBAL_BATCH_SIZE=0):
        # X = tf.expand_dims(inputs[4], -1)
        X = inputs[4]
        y_r_ar = tf.expand_dims(inputs[2], -1)
        y_r_val = tf.expand_dims(inputs[3], -1)
        print(X)
        _, prediction_ar, prediction_val, _ = student_model(X, training=False)

        return prediction_ar, prediction_val, y_r_ar, y_r_val

with strategy.scope():
    # `experimental_run_v2` replicates the provided computation and runs it
    # with the distributed input.

    @tf.function
    def distributed_test_step_teacher(dataset_inputs, GLOBAL_BATCH_SIZE):
        per_replica_losses = strategy.run(test_step_teacher,
                                          args=(dataset_inputs, GLOBAL_BATCH_SIZE))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


    @tf.function
    def distributed_test_step_student(dataset_inputs, GLOBAL_BATCH_SIZE):
        per_replica_losses = strategy.run(test_step_student,
                                          args=(dataset_inputs, GLOBAL_BATCH_SIZE))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


    ar_teacher_results = []
    val_teacher_results = []
    for step, test in enumerate(test_data_teacher):
        prediction_ar, prediction_val, y_r_ar, y_r_val = distributed_test_step_teacher(test, data_fetch_teacher.test_n)
        # print(template.format(prediction_ar.numpy()[0, 0],  prediction_val.numpy()[0, 0], y_r_ar.numpy()[0, 0], y_r_val.numpy()[0, 0]))
        ar_teacher_results.append([prediction_ar.numpy()[0, 0], y_r_ar.numpy()[0, 0]])
        val_teacher_results.append([prediction_val.numpy()[0, 0], y_r_val.numpy()[0, 0]])

    ar_teacher_results = np.array(ar_teacher_results)
    val_teacher_results = np.array(val_teacher_results)

    ar_student_results = []
    val_student_results = []
    for step, test in enumerate(test_data_student):
        prediction_ar, prediction_val, y_r_ar, y_r_val = distributed_test_step_student(test, data_fetch_student.test_n)
        # print(template.format(prediction_ar.numpy()[0, 0],  prediction_val.numpy()[0, 0], y_r_ar.numpy()[0, 0], y_r_val.numpy()[0, 0]))
        ar_student_results.append([prediction_ar.numpy()[0, 0], y_r_ar.numpy()[0, 0]])
        val_student_results.append([prediction_val.numpy()[0, 0], y_r_val.numpy()[0, 0]])

    ar_student_results = np.array(ar_student_results)
    val_student_results = np.array(val_student_results)

    features_file = pd.read_csv(testing_data)
    features_file["Valence_label"] = val_teacher_results[:, 1]
    features_file["Arousal_label"] = ar_teacher_results[:, 1]
    features_file["Valence_pred_teacher"] = val_teacher_results[:, 0]
    features_file["Arousal_pred_teacher"] = ar_teacher_results[:, 0]
    # features_file["Valence_label_student"] = val_student_results[:, 1]
    # features_file["Arousal_label_student"] = ar_student_results[:, 1]
    features_file["Valence_pred_student"] = val_student_results[:, 0]
    features_file["Arousal_pred_student"] = ar_student_results[:, 0]

    features_file.to_csv(TRAINING_RESULTS_PATH + "TimeSeriesResult_" + subject + "-" + date + ".csv")
    # features_file.to_csv(TRAINING_RESULTS_PATH + "TimeSeriesResult_all.csv")
