import tensorflow as tf
from KnowledgeDistillation.Models.EnsembleDistillModel import EnsembleStudentOneDim
from KnowledgeDistillation.Models.EnsembleFeaturesModel import EnsembleSeparateModel
from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, TRAINING_RESULTS_PATH, ROAD_ECG, SPLIT_TIME, STRIDE, ECG_N
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetch, DataFetchRoad
import datetime
import os
import sys

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
num_output_ar = 3
num_output_val = 3
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
checkpoint_prefix = result_path + "model_student"

# datagenerator
testing_data = DATASET_PATH + "\\stride=0.2\\test_data_" + str(fold) + ".csv"
data_fetch = DataFetch(test_file=testing_data,
                       ECG_N=ECG_RAW_N, KD=True, multiple=True, training=False)
generator = data_fetch.fetch



test_generator = tf.data.Dataset.from_generator(
    lambda: generator(),
    output_types=(tf.float32),
    output_shapes=(tf.TensorShape([ECG_RAW_N])))

test_data = test_generator.batch(BATCH_SIZE)

with strategy.scope():
    # load pretrained model
    # encoder model

    model = EnsembleStudentOneDim(num_output_ar=num_output_ar, num_output_val=num_output_val)
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                   decay_steps=EPOCHS, decay_rate=0.95,
                                                                   staircase=True)
    optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)



# Manager
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, base_model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
checkpoint.restore(manager.latest_checkpoint)

predictions = []
print("Start testing")
with strategy.scope():


    def test_step(inputs, GLOBAL_BATCH_SIZE=0):
        X = tf.expand_dims(tf.expand_dims(inputs[-1], -1), 0)
        y_r_ar = tf.expand_dims(inputs[3], -1)
        y_r_val = tf.expand_dims(inputs[4], -1)

        prediction_ar, prediction_val = model.predict(X, global_batch_size=GLOBAL_BATCH_SIZE, training=False)

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

    template = ("{}, {}, {}, {}")
    for step, test in enumerate(test_data):
        prediction_ar, prediction_val, y_r_ar, y_r_val = distributed_test_step(test, data_fetch.test_n)
        print(template.format(prediction_ar.numpy()[0, 0],  prediction_val.numpy()[0, 0], y_r_ar.numpy()[0, 0], y_r_val.numpy()[0, 0]))


