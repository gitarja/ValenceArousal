import tensorflow as tf
import numpy as np
from KnowledgeDistillation.Models.EnsembleFeaturesModel import EnsembleSeparateModel, EnsembleModel
from Conf.Settings import ECG_RAW_N, TRAINING_RESULTS_PATH, ROAD_ECG, SPLIT_TIME, STRIDE, ECG_N, N_CLASS, TRAINING_ECG, EXTENTION_TIME
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetchRoadTimeSeries
from Libs.Utils import regressLabelRoad
import pandas as pd
import os
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# def convertBond(x):
#     if x < -2:
#         return -2
#     if x > 2:
#         return 2
#     return x

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
fold = 4
prev_val_loss = 1000
wait_i = 0
result_path = TRAINING_RESULTS_PATH + "Binary_ECG\\fold_" + str(fold) + "\\"
checkpoint_prefix = "D:\\usr\\nishihara\\result\\ValenceArousal\\Binary_ECG\\fromPras\\model_student_ECG_KD\\"

# datagenerator
for subject_path in glob.glob(TRAINING_ECG + "*\\"):
    print(subject_path)
    ecg = glob.glob(subject_path + "*_HB_PW.csv")
    if len(ecg) > 0:
        ecg_data = ecg[0]
        data_fetch = DataFetchRoadTimeSeries(ecg_file=ecg_data, stride=STRIDE, ecg_n=ECG_RAW_N, split_time=SPLIT_TIME)
        generator = data_fetch.fetch

        test_generator = tf.data.Dataset.from_generator(
            lambda: generator(),
            output_types=(tf.string, tf.float32, tf.float32),
            output_shapes=((), (), tf.TensorShape([ECG_N])))

        test_data = test_generator.batch(BATCH_SIZE)

        with strategy.scope():
            # load model
            model = EnsembleModel(num_output=num_output).loadBaseModel(checkpoint_prefix)

        predictions = []
        print("Start testing")

        with strategy.scope():
            def test_step(inputs, GLOBAL_BATCH_SIZE=0):
                X = inputs[-1]
                # X = tf.expand_dims(inputs[-1], 0)
                _, prediction_ar, prediction_val, _ = model(X, training=False)
                # _, prediction_ar, prediction_val = model(X, training=False)

                return prediction_ar, prediction_val

        with strategy.scope():
            # `experimental_run_v2` replicates the provided computation and runs it
            # with the distributed input.

            @tf.function
            def distributed_test_step(dataset_inputs, GLOBAL_BATCH_SIZE):
                per_replica_losses = strategy.run(test_step,
                                                  args=(dataset_inputs, GLOBAL_BATCH_SIZE))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


            it = 0

            data = pd.DataFrame(columns=["time", "arousal", "valence", "color"])
            for step, test in enumerate(test_data):
                timestamp = test[0]
                time = test[1]
                prediction_ar, prediction_val = distributed_test_step(test, BATCH_SIZE)
                arousal = np.max([np.min([prediction_ar.numpy()[0, 0], 2]), -2])
                valence = np.max([np.min([prediction_val.numpy()[0, 0], 2]), -2])
                if len(data) > 0:
                    tdelta = time.numpy()[0] - data.iloc[-1]["time"]
                    if tdelta > (SPLIT_TIME * STRIDE + EXTENTION_TIME):
                        for j in np.arange(0, (tdelta // (SPLIT_TIME * STRIDE))):
                            time_zero = data.iloc[-1]["time"] + (SPLIT_TIME * STRIDE)
                            data = data.append({"time": time_zero,
                                                "arousal": 0,
                                                "valence": 0,
                                                "color": 0},
                                               ignore_index=True)

                data = data.append({"time": time.numpy()[0],
                                    "arousal": arousal,
                                    "valence": valence,
                                    "color": regressLabelRoad(arousal, valence, th=th)},
                                   ignore_index=True)

            results_file_name = subject_path + "results_timeseries.csv"
            data.to_csv(results_file_name, index=False)
            # print(template.format(convertBond(prediction_ar.numpy()[0, 0]),  convertBond(prediction_val.numpy()[0, 0])))


