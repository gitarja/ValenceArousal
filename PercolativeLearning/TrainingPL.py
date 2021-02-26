import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from PercolativeLearning.PLModel import PercolativeLearning
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, Precision, Recall
from tensorflow.keras.utils import plot_model, to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Libs.Utils import arToLabels, valToLabels, arValMulLabels, TrainingHistory

LEARNING_RATE = 0.001
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
BATCH_SIZE = 512
EPOCHS_PREV = 1000
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
NUM_CLASSES = 4

STRIDE = 0.1
eda_features = []
ppg_features = []
resp_features = []
eeg_features = []
ecg_features = []
ecg_resp_features = []
Y_arousal = []
Y_valence = []

data_path = "G:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\2020-*"

# load features data
print("Loading data...")
for count, folder in enumerate(glob.glob(data_path)):
    print("{}/{}".format(count + 1, len(glob.glob(data_path))) + "\r", end="")
    for subject in glob.glob(folder + "\\*-2020-*"):
        eeg_path = subject + "\\results_stride=" + str(STRIDE) + "\\EEG\\"
        eda_path = subject + "\\results_stride=" + str(STRIDE) + "\\eda\\"
        ppg_path = subject + "\\results_stride=" + str(STRIDE) + "\\ppg\\"
        resp_path = subject + "\\results_stride=" + str(STRIDE) + "\\Resp\\"
        ecg_path = subject + "\\results_stride=" + str(STRIDE) + "\\ECG\\"
        ecg_resp_path = subject + "\\results_stride=" + str(STRIDE) + "\\ECG_resp\\"

        features_list = pd.read_csv(subject + "\\features_list_" + str(STRIDE) + ".csv")
        features_list["Valence"] = features_list["Valence"].apply(valToLabels)
        features_list["Arousal"] = features_list["Arousal"].apply(arToLabels)
        for i in range(len(features_list)):
            filename = features_list.iloc[i]["Idx"]
            eda_features.append(np.load(eda_path + "eda_" + str(filename) + ".npy"))
            ppg_features.append(np.load(ppg_path + "ppg_" + str(filename) + ".npy"))
            resp_features.append(np.load(resp_path + "resp_" + str(filename) + ".npy"))
            eeg_features.append(np.load(eeg_path + "eeg_" + str(filename) + ".npy"))
            ecg_features.append(np.load(ecg_path + "ecg_" + str(filename) + ".npy"))
            ecg_resp_features.append(np.load(ecg_resp_path + "ecg_resp_" + str(filename) + ".npy"))
            Y_arousal.append(features_list.iloc[i]["Arousal"])
            Y_valence.append(features_list.iloc[i]["Valence"])

eda_features = np.array(eda_features)
ppg_features = np.array(ppg_features)
resp_features = np.array(resp_features)
eeg_features = np.array(eeg_features)
ecg_features = np.array(ecg_features)
ecg_resp_features = np.array(ecg_resp_features)
X_main = np.concatenate([ecg_features], axis=1)
X_aux = np.concatenate([eda_features, ppg_features, resp_features, ecg_resp_features, eeg_features], axis=1)
Y = [arValMulLabels(ar, val) for ar, val in zip(Y_arousal, Y_valence)]
input_dim_main = X_main.shape[1]
input_dim_aux = X_aux.shape[1]
input_dim_alpha = 1
num_data = X_main.shape[0]

# Transform Y to one-hot vector
Y = to_categorical(Y, NUM_CLASSES)

# Standardize X_main and X_aux
ss = MinMaxScaler()
X_main = ss.fit_transform(X_main)
X_aux = ss.fit_transform(X_aux)

# Split train, validation and test
X_main_train, X_main_val, X_main_test = np.split(X_main, [-int(num_data * (VALIDATION_SPLIT + TEST_SPLIT)),
                                                          -int(num_data * TEST_SPLIT)], axis=0)
X_aux_train, X_aux_val, X_aux_test = np.split(X_aux, [-int(num_data * (VALIDATION_SPLIT + TEST_SPLIT)),
                                                      -int(num_data * TEST_SPLIT)], axis=0)
Y_train, Y_val, Y_test = np.split(Y, [-int(num_data * (VALIDATION_SPLIT + TEST_SPLIT)), -int(num_data * TEST_SPLIT)],
                                  axis=0)

print("Input dim: main: {}, aux: {}".format(X_main_train.shape[1], X_aux_train.shape[1]))
print("Num of data: Train: {}, Validation: {}, Test: {}".format(X_main_train.shape[0], X_main_val.shape[0], X_main_test.shape[0]))

# Make dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_main_train, X_aux_train, Y_train)).shuffle(num_data).batch(
    BATCH_SIZE)
validation_dataset = tf.data.Dataset.from_tensor_slices((X_main_val, X_aux_val, Y_val)).shuffle(num_data).batch(
    BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_main_test, X_aux_test, Y_test)).shuffle(num_data).batch(BATCH_SIZE)

# Define Model
input_main = Input(shape=(input_dim_main,))
input_aux = Input(shape=(input_dim_aux,))
input_alpha = Input(shape=(input_dim_alpha,))
PL = PercolativeLearning(num_classes=NUM_CLASSES)

feature = PL.createPercNet(input_main, input_aux, input_alpha)
perc_network = Model(inputs=[input_main, input_aux, input_alpha], outputs=feature)
logit = PL.createIntNet(input=perc_network.output)
whole_network = Model(inputs=[input_main, input_aux, input_alpha], outputs=logit)
# perc_network.compile(optimizer=OPTIMIZER)
# whole_network.compile(optimizer=OPTIMIZER, loss_epochs=CategoricalCrossentropy(), metrics=CategoricalAccuracy())
perc_network.summary()
whole_network.summary()
plot_model(perc_network, to_file="perc_model.png", show_shapes=True)
plot_model(whole_network, to_file="whole_model.png", show_shapes=True)

loss_metrics = CategoricalCrossentropy(from_logits=True)


# Define loss_epochs and gradient
@tf.function
def computeLoss(model: Model, x_main, x_aux, alpha, y_true):
    y_pred = model([x_main, x_aux, alpha])
    return loss_metrics(y_true, y_pred)


@tf.function
def computeGradient(model: Model, x_main, x_aux, alpha, y_true):
    with tf.GradientTape() as tape:
        loss_value = computeLoss(model, x_main, x_aux, alpha, y_true)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


train_result = TrainingHistory()
validation_result = TrainingHistory()

# Previous training
alpha = np.ones(1, dtype="float32")
for epoch in range(EPOCHS_PREV):
    epoch_loss_train = Mean()
    epoch_accuracy_train = CategoricalAccuracy()
    epoch_loss_val = Mean()
    epoch_accuracy_val = CategoricalAccuracy()

    for (x_main_train, x_aux_train, y_train), (x_main_val, x_aux_val, y_val) in tf.data.Dataset.zip(
            (train_dataset, validation_dataset)):
        loss_value_train, grad = computeGradient(whole_network, x_main_train, x_aux_train, alpha, y_train)
        OPTIMIZER.apply_gradients(zip(grad, whole_network.trainable_variables))
        y_train_pred = tf.nn.softmax(whole_network([x_main_train, x_aux_train, alpha]))
        epoch_loss_train(loss_value_train)
        epoch_accuracy_train(y_train, y_train_pred)

        loss_value_val = computeLoss(whole_network, x_main_val, x_aux_val, alpha, y_val)
        y_val_pred = tf.nn.softmax(whole_network([x_main_val, x_aux_val, alpha]))
        epoch_loss_val(loss_value_val)
        epoch_accuracy_val(y_val, y_val_pred)

    train_result.loss_epochs.append(epoch_loss_train.result().numpy())
    train_result.acc_epochs.append(epoch_accuracy_train.result().numpy())
    validation_result.loss_epochs.append(epoch_loss_val.result().numpy())
    validation_result.acc_epochs.append(epoch_accuracy_val.result().numpy())

    print(
        "Epoch {}/{} Train Loss: {:.3f}, Train Accuracy: {:.3%}, Validation Loss: {:.3f}, Validation Accuracy: {:.3%}, ".format(
            epoch + 1, EPOCHS_PREV,
            train_result.loss_epochs[-1],
            train_result.acc_epochs[-1],
            validation_result.loss_epochs[-1],
            validation_result.acc_epochs[-1]))



