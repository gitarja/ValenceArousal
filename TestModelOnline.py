import tensorflow as tf
from KnowledgeDistillation.Models.EnsembleDistillModel import EnsembleStudentOneDim
from KnowledgeDistillation.Models.EnsembleFeaturesModel import EnsembleSeparateModel, EnsembleModel
from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, TRAINING_RESULTS_PATH, ROAD_ECG, SPLIT_TIME, STRIDE, ECG_N, N_CLASS
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetch, DataFetchRoad
import datetime
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_style("whitegrid")
sns.set_color_codes("dark")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
EPOCHS = 200
PRE_EPOCHS = 100
BATCH_SIZE = 1
th = 0.5
ALL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
wait = 10

# setting
# fold = str(sys.argv[1])
fold=1
prev_val_loss = 1000
wait_i = 0
result_path = TRAINING_RESULTS_PATH + "Binary_ECG\\fold_" + str(fold) + "\\"
checkpoint_prefix = result_path + "model_student_ECG_KD_high"
checkpoint_prefix2 = result_path + "model_student_ECG_KD"
# datagenerator
testing_data = DATASET_PATH + "\\stride=0.2\\test_data_" + str(fold) + ".csv"
validation_data = DATASET_PATH + "\\stride=0.2\\validation_data_" + str(fold) + ".csv"
data_fetch = DataFetch(test_file=testing_data, validation_file=validation_data,
                       ECG_N=ECG_RAW_N, KD=True, training=False, teacher=False, ECG=True, high_only=False)
generator = data_fetch.fetch



test_generator = tf.data.Dataset.from_generator(
    lambda: generator(training_mode=2),
    # output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
    # output_shapes=(tf.TensorShape([FEATURES_N]), (tf.TensorShape([N_CLASS])), (), (), tf.TensorShape([ECG_RAW_N]), ()))
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([FEATURES_N]), (tf.TensorShape([N_CLASS])), (), (), tf.TensorShape([ECG_N]), ()))

test_data = test_generator.batch(BATCH_SIZE)

with strategy.scope():
    # load pretrained model
    # encoder model

    # model = EnsembleStudentOneDim(num_output=num_output)
    model = EnsembleModel(num_output=num_output).loadBaseModel(checkpoint_prefix=checkpoint_prefix2)





predictions = []
print("Start testing")
with strategy.scope():


    def test_step(inputs, GLOBAL_BATCH_SIZE=0):
        # X = tf.expand_dims(inputs[4], -1)
        X = inputs[4]
        y_r_ar = tf.expand_dims(inputs[2], -1)
        y_r_val = tf.expand_dims(inputs[3], -1)
        print(X)
        _, prediction_ar, prediction_val, _ = model(X, training=False)


        return prediction_ar, prediction_val, y_r_ar, y_r_val




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
    ar_results = []
    val_results = []
    template = ("{}, {}, {}, {}")
    for step, test in enumerate(test_data):
        prediction_ar, prediction_val, y_r_ar, y_r_val = distributed_test_step(test, data_fetch.test_n)
        # print(template.format(prediction_ar.numpy()[0, 0],  prediction_val.numpy()[0, 0], y_r_ar.numpy()[0, 0], y_r_val.numpy()[0, 0]))
        ar_results.append([prediction_ar.numpy()[0, 0], y_r_ar.numpy()[0, 0]])
        val_results.append([prediction_val.numpy()[0, 0], y_r_val.numpy()[0, 0]])

    ar_results = np.array(ar_results)

    val_results = np.array(val_results)

    th = .5
    # ambigous
    ar_a_v_a_results = np.average(((np.abs(ar_results[:, 0]) <= th) | (np.abs(val_results[:, 0]) <= th)) & ((np.abs(ar_results[:, 1]) == 0) | (np.abs(val_results[:, 1]) == 0)))
    ar_na_v_na_results = np.average(((np.abs(ar_results[:, 0]) > th) | (np.abs(val_results[:, 0]) > th)) & (((np.abs(ar_results[:, 1]) > 0) | (np.abs(val_results[:, 1]) > 0))))

    ar_p_v_p = (ar_results[:, 1] > th) & (val_results[:, 1] > th)
    ar_p_v_p_results = np.average((ar_results[ar_p_v_p, 0] > 0.) & (val_results[ar_p_v_p, 0] > 0))

    # ar positif and val negatif
    ar_p_v_n = (ar_results[:, 1] > th) & (val_results[:, 1] < -th)
    ar_p_v_n_results = np.average((ar_results[ar_p_v_n, 0] > 0) & (val_results[ar_p_v_n, 0] < -0))

    # ar negatif and val positif
    ar_n_v_p = (ar_results[:, 1] < -th) & (val_results[:, 1] > th)
    ar_n_v_p_results = np.average((ar_results[ar_n_v_p, 0] < -0) & (val_results[ar_n_v_p, 0] > 0))

    # ar negatif and val negatif
    ar_n_v_n = (ar_results[:, 1] < -th) & (val_results[:, 1] < -th)
    ar_n_v_n_results = np.average((ar_results[ar_n_v_n, 0] < -0) & (val_results[ar_n_v_n, 0] < -0))

    # val positif
    a_p = (ar_results[:, 1] > 0)
    a_n = (ar_results[:, 1] < 0)
    a_p_results = np.sum(ar_results[a_p, 0] > 0)
    a_n_results = np.sum(ar_results[a_n, 0] <= 0)
    print((a_p_results + a_n_results) / (np.sum(a_p) + np.sum(a_n)))
    #val positif
    v_p = (val_results[:, 1] > 0)
    v_n = (val_results[:, 1] < 0)
    v_p_results = np.sum(val_results[v_p, 0] > 0)
    v_n_results = np.sum(val_results[v_n, 0] <= 0)
    print((v_p_results + v_n_results) / (np.sum(v_p) + np.sum(v_n)))

    print(str(ar_p_v_p_results) + "," + str(ar_p_v_n_results) + "," + str(ar_n_v_p_results) + "," + str(ar_n_v_n_results))
    print(str(ar_a_v_a_results) + ","+ str(ar_na_v_na_results))
    #plotting

    # plt.figure(1)
    # plt.plot(ar_results[:, 0], '-', linewidth=1.5, label="Arousal-prediction", color="#66c2a5")
    # plt.plot(ar_results[:, 1], '-', linewidth=1.5, label="Arousal-label", color="#fc8d62")
    # plt.legend()
    # plt.savefig("arousal.png")
    # plt.figure(2)
    # plt.plot(val_results[:, 0], '-', linewidth=1.5,  label="Valence-prediction", color="#66c2a5")
    # plt.plot(val_results[:, 1], '-', linewidth=1.5, label="Valence-label", color="#fc8d62")
    # plt.legend()
    # plt.savefig("valence.png")
    # plt.show()


