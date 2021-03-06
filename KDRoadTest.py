import tensorflow as tf
from KnowledgeDistillation.Models.EnsembleDistillModel import EnsembleStudentOneDim
from KnowledgeDistillation.Models.EnsembleFeaturesModel import EnsembleSeparateModel, EnsembleModel
from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, TRAINING_RESULTS_PATH, ROAD_ECG, SPLIT_TIME, STRIDE, ECG_N, N_CLASS
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetch, DataFetchRoad
import datetime
import pandas as pd
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def convertBond(x):
    if x < -2:
        return -2
    if x > 2:
        return 2
    return x

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
fold=4
prev_val_loss = 1000
wait_i = 0
result_path = TRAINING_RESULTS_PATH + "Binary_ECG\\fold_" + str(fold) + "\\"
checkpoint_prefix = result_path + "model_student_pre_KD"

# datagenerator
results_file_name = ROAD_ECG + "TS103_training\\20200617_141440_279_results.csv"
ecg_data = ROAD_ECG + "TS103_training\\20200617_141440_279_HB_PW.csv"
gps_data = ROAD_ECG + "TS103_training\\20200617_141440_279_GPS.csv"
mask_data = ROAD_ECG + "E5\\20201027_161000_536_HB_PW.csv"
data_fetch = DataFetchRoad(ecg_file=ecg_data, gps_file=gps_data, mask_file=mask_data, stride=STRIDE, ecg_n=ECG_RAW_N, split_time=SPLIT_TIME)
generator = data_fetch.fetch



test_generator = tf.data.Dataset.from_generator(
    lambda: generator(),
    output_types=(tf.float32),
    output_shapes=(tf.TensorShape([ECG_N])))

test_data = test_generator.batch(BATCH_SIZE)

with strategy.scope():
    # load pretrained model
    # encoder model
    checkpoint_prefix_encoder = result_path + "model_base_student"
    model = EnsembleStudentOneDim(num_output=num_output)
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                   decay_steps=EPOCHS, decay_rate=0.95,
                                                                   staircase=True)
    optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)



# Manager
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), student_model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
checkpoint.restore(manager.latest_checkpoint)

predictions = []
print("Start testing")
with strategy.scope():


    def test_step(inputs, GLOBAL_BATCH_SIZE=0):
        X = tf.expand_dims(tf.expand_dims(inputs[-1], -1), 0)
        # X = tf.expand_dims(inputs[-1], 0)
        prediction_ar, prediction_val = model.predict(X, global_batch_size=GLOBAL_BATCH_SIZE, training=False)
        # _, prediction_ar, prediction_val = model(X, training=False)

        return prediction_ar, prediction_val




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

    template = ("{}, {}")
    data = pd.DataFrame(columns=["arousal", "valence"])
    for step, test in enumerate(test_data):
        prediction_ar, prediction_val = distributed_test_step(test, data_fetch.test_n)
        data = data.append({'arousal': prediction_ar.numpy()[0, 0], 'valence': prediction_val.numpy()[0, 0]}, ignore_index=True)

    data.to_csv(results_file_name)
        # print(template.format(convertBond(prediction_ar.numpy()[0, 0]),  convertBond(prediction_val.numpy()[0, 0])))


