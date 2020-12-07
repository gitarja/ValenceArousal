from KnowledgeDistillation.Models.EnsembleDistillModel import Baseline, EnsembleStudent, EnsembleTeacher
import tensorflow as tf
from KnowledgeDistillation.Utils.DataGenerator import DataGenerator

# folder
path = "D:\\usr\\pras\\data\\EmotionTestVR\\"
training_file = "D:\\usr\\pras\\data\\EmotionTestVR\\training.csv"
testing_file = "D:\\usr\\pras\\data\\EmotionTestVR\\testing.csv"

# model param
N = 1945
output = 2
batch_size = 64
val_batch = 51
learning_rate_ensemble = 0.095e-3
learning_rate_baseline = 0.095e-3
EPOCH_NUM = 100
# setting generator
ecg_length = 11236
ecg_dim = 106
data_fetch = DataGenerator(path=path, training_list_file=training_file,
                           testing_list_file=testing_file, ecg_length=ecg_length, batch_size=batch_size, transform=True)
generator = data_fetch.fetchData

# setting model
model_ensemble = EnsembleTeacher(num_output=1)
model_student = EnsembleStudent(num_output=1)
model_baseline = Baseline(num_output=1)

# generator

train_data = tf.data.Dataset.from_generator(
    lambda: generator(),
    output_types=(tf.float32, tf.float32, tf.int32, tf.int32),
    output_shapes=(tf.TensorShape([N]), tf.TensorShape([ecg_dim, ecg_dim, 1]), (), ()))

val_data = tf.data.Dataset.from_generator(
    lambda: generator(training=False),
    output_types=(tf.float32, tf.float32, tf.int32, tf.int32),
    output_shapes=(tf.TensorShape([N]), tf.TensorShape([ecg_dim, ecg_dim, 1]), (), ()))

# loss

cross_loss = tf.losses.BinaryCrossentropy(from_logits=True)
# optimizer
# learning_rate_ensemble = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate_ensemble,
#                                                                decay_steps=EPOCH_NUM, decay_rate=0.95, staircase=True)
# learning_rate_baseline = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate_baseline,
#                                                                decay_steps=EPOCH_NUM, decay_rate=0.95, staircase=True)
optimizer_ensemble = tf.keras.optimizers.Adamax(learning_rate=learning_rate_ensemble)
optimizer_baseline = tf.keras.optimizers.Adamax(learning_rate=learning_rate_baseline)

# metrics
loss_ensemble_metric = tf.keras.metrics.Mean()
loss_baseline_metric = tf.keras.metrics.Mean()
loss_student_metric = tf.keras.metrics.Mean()

# accuracy
# acc_ensemble = tf.keras.metrics.Accuracy()
# acc_baseline = tf.keras.metrics.Accuracy()

acc_ensemble = tf.keras.metrics.BinaryAccuracy()
acc_baseline = tf.keras.metrics.BinaryAccuracy()
acc_student = tf.keras.metrics.BinaryAccuracy()

prec_ensemble = tf.keras.metrics.Precision()
prec_baseline = tf.keras.metrics.Precision()
prec_student = tf.keras.metrics.Precision()

rec_ensemble = tf.keras.metrics.Recall()
rec_baseline = tf.keras.metrics.Recall()
rec_student = tf.keras.metrics.Recall()

train_data = train_data.shuffle(data_fetch.len_train).padded_batch(batch_size, padded_shapes=(
    tf.TensorShape([N]), tf.TensorShape([ecg_dim, ecg_dim, 1]), (), ()))

val_data = val_data.padded_batch(val_batch, padded_shapes=(
    tf.TensorShape([N]), tf.TensorShape([ecg_dim, ecg_dim, 1]), (), ()))

for epoch in range(EPOCH_NUM):
    for step, inputs in enumerate(train_data):
        X, ecg, y_val, y_ar = inputs
        with tf.GradientTape() as tape_ensemble, tf.GradientTape() as tape_student:
            z_ensemble_val, z_ensemble_ar = model_ensemble(X, training=True)
            z_student_val, z_student_ar = model_student(ecg, training=True)
            loss_ensemble = 0.5 * (cross_loss(y_val, z_ensemble_val) + cross_loss(y_ar, z_ensemble_ar))
            loss_student =  0.5*(cross_loss(y_val, z_student_val) + cross_loss(y_ar, z_student_ar))
        grads_ensemble = tape_ensemble.gradient(loss_ensemble, model_ensemble.trainable_weights)
        grads_student = tape_student.gradient(loss_student, model_student.trainable_weights)
        optimizer_ensemble.apply_gradients(zip(grads_ensemble, model_ensemble.trainable_weights))
        optimizer_ensemble.apply_gradients(zip(grads_student, model_student.trainable_weights))

        with tf.GradientTape() as tape_baseline:
            z_baseline_val, z_baseline_ar = model_baseline(X, training=True)
            loss_baseline = 0.5 * (cross_loss(y_val, z_baseline_val) + cross_loss(y_ar, z_baseline_ar))
        grads_baseline = tape_baseline.gradient(loss_baseline, model_baseline.trainable_weights)
        optimizer_baseline.apply_gradients(zip(grads_baseline, model_baseline.trainable_weights))

    for step_val, val_inputs in enumerate(val_data):
        X, ecg, y_val, y_ar = val_inputs

        #ensemble
        z_ensemble_val, z_ensemble_ar = model_ensemble(X, training=False)
        predictions_ensemble = tf.concat([tf.nn.sigmoid(z_ensemble_val), tf.nn.sigmoid(z_ensemble_ar)], 0)
        y = tf.concat([y_val, y_ar], 0)
        loss_ensemble = 0.5 * (cross_loss(y_val, z_ensemble_val) + cross_loss(y_ar, z_ensemble_ar))
        loss_ensemble_metric(loss_ensemble)
        acc_ensemble(y, predictions_ensemble)
        prec_ensemble(y, predictions_ensemble)
        rec_ensemble(y, predictions_ensemble)


        #student
        z_student_val, z_student_ar = model_student(ecg, training=False)
        predictions_student = tf.concat([tf.nn.sigmoid(z_student_val), tf.nn.sigmoid(z_student_ar)], 0)
        y = tf.concat([y_val, y_ar], 0)
        loss_student = 0.5 * (cross_loss(y_val, z_student_val) + cross_loss(y_ar, z_student_ar))

        loss_student_metric(loss_student)
        acc_student(y, predictions_student)
        prec_student(y, predictions_student)
        rec_student(y, predictions_student)
        # print(y)
        # print(predictions_ensemble)

        z_baseline_val, z_baseline_ar = model_baseline(X, training=False)
        predictions_baseline = tf.concat([tf.nn.sigmoid(z_baseline_val), tf.nn.sigmoid(z_baseline_ar)], 0)
        loss_baseline = 0.5 * (cross_loss(y_val, z_baseline_val) + cross_loss(y_ar, z_baseline_ar))
        loss_baseline_metric(loss_baseline)
        acc_baseline(y, predictions_baseline)
        prec_baseline(y, predictions_baseline)
        rec_baseline(y, predictions_baseline)


    # template = (
    #     "Epoch {}, Loss En: {:.4f}, Acc En: {:.2f}, Prec En: {:.2f}, Rec En: {:.2f}, Loss Bas: {:.4f}, Acc Bas: {:.2f}, Prec Bas: {:.2f}, Rec Bas: {:.2f}")
    # print(template.format(epoch + 1, loss_ensemble_metric.result().numpy(), acc_ensemble.result().numpy(),
    #                       prec_ensemble.result().numpy(), rec_ensemble.result().numpy(),
    #                       loss_baseline_metric.result().numpy(), acc_baseline.result().numpy(), prec_baseline.result().numpy(), rec_baseline.result().numpy()))


    template = (
        "Epoch {}, Loss En: {:.4f}, Acc En: {:.2f}, Prec En: {:.2f}, Rec En: {:.2f}, Loss Stud: {:.4f}, Acc Stud: {:.2f}, Prec Stud: {:.2f}, Rec Stud: {:.2f}")
    print(template.format(epoch + 1, loss_ensemble_metric.result().numpy(), acc_ensemble.result().numpy(),
                          prec_ensemble.result().numpy(), rec_ensemble.result().numpy(),
                          loss_student_metric.result().numpy(), acc_student.result().numpy(), prec_student.result().numpy(), rec_student.result().numpy()))

    loss_ensemble_metric.reset_states()
    loss_baseline_metric.reset_states()
    loss_student_metric.reset_states()

    acc_ensemble.reset_states()
    acc_baseline.reset_states()
    acc_student.reset_states()

    prec_ensemble.reset_states()
    prec_baseline.reset_states()
    prec_student.reset_states()

    rec_ensemble.reset_states()
    rec_baseline.reset_states()
    rec_student.reset_states()
