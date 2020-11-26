import tensorflow as tf
from KnowledgeDistillation.Models.EnsembleFeaturesModel import EnsembleSeparateModel

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
num_output = 1
initial_learning_rate = 1e-3
EPOCHS = 300
BATCH = 10
GLOBAL_BATCH_SIZE = BATCH * strategy.num_replicas_in_sync

# generate dummy data
N = 20
X = tf.random.uniform(shape=(N, 100))
y = tf.cast(tf.transpose(tf.random.categorical(tf.math.log([[0.5, 0.5]]), N)), tf.dtypes.float32)

X_tensor = tf.data.Dataset.from_tensor_slices(X)
y_tensor = tf.data.Dataset.from_tensor_slices(y)

train_dataset = tf.data.Dataset.zip((X_tensor, y_tensor)).repeat(3).batch(BATCH)

with strategy.scope():
    model = EnsembleSeparateModel(num_output=num_output)

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                   decay_steps=EPOCHS, decay_rate=0.95, staircase=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

    # ---------------------------Epoch&Loss--------------------------#
    loss_metric = tf.keras.metrics.Mean()



with strategy.scope():
    def train_step(inputs):
        X = inputs[0]
        y = inputs[1]


        with tf.GradientTape() as tape:
            losses, org_losses = model.trainSMCL(X, y, GLOBAL_BATCH_SIZE)

        # update gradient
        grads = tape.gradient(losses, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        loss_metric(org_losses)
        return losses

with strategy.scope():
    # `experimental_run_v2` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step,
                                          args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


    for epoch in range(EPOCHS):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for step, train in enumerate(train_dataset):
            distributed_train_step(train)

        template = ("Epoch {}, Loss: {}")
        print(template.format(epoch + 1, loss_metric.result().numpy()))

        loss_metric.reset_states()
