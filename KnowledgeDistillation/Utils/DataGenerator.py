import pandas as pd
import numpy as np
from joblib import dump, load
from scipy import interpolate
class DataGenerator():

    def __init__(self, path, training_list_file, testing_list_file, val_th=3, ar_th=3, ecg_length=10000, batch_size=32, transform=False):
        '''
        :param path: directory path for teacher
        :param teacher_list_file: list features for training
        :param testing_list_file: list features for testing
        :param val_th: threshold deciding the valence valence
        :param ar_th: threshold deciding the arousal valence
        '''
        self.path = path


        self.eeg_path = "EEG\\"
        self.gsr_path = "GSR\\"
        self.resp_path = "Resp\\"
        self.ecg_path = "ECG\\"

        self.ecg_raw_path = "ECG_raw\\"
        self.ecg_length = ecg_length

        self.train_data = pd.read_csv(training_list_file)
        self.test_data = pd.read_csv(testing_list_file)


        self.batch_size = batch_size
        self.len_train = len(self.train_data)
        self.transform = transform
        if transform:
            self.scaler = load('scaler.joblib')




    def fetchData(self, training=True, split=True):
        i = 0
        if training:
            data = self.train_data
        else:
            data = self.test_data

        while i < len(data):
            row = data.iloc[i]
            filename = row["Idx"]
            val_label = row["Valence"]
            ar_label = row["Arousal"]
            subject = row["Subject"]
            subject_path = self.path + subject + "\\results\\"
            eda_features = np.load(subject_path + self.gsr_path + "eda_" + str(filename) + ".npy")
            ppg_features = np.load(subject_path + self.gsr_path + "ppg_" + str(filename) + ".npy")
            resp_features = np.load(subject_path + self.resp_path + "resp_" + str(filename) + ".npy")
            eeg_features = np.load(subject_path + self.eeg_path + "eeg_" + str(filename) + ".npy")
            ecg_features = np.load(subject_path + self.ecg_path + "ecg_" + str(filename) + ".npy")

            ecg_raw = np.load(subject_path + self.ecg_raw_path + "ecg_raw_" + str(filename) + ".npy")
            if len(ecg_raw) < self.ecg_length:
                f = interpolate.interp1d(np.arange(0, len(ecg_raw)), ecg_raw)
                xnew = np.arange(0, len(ecg_raw) - 1, len(ecg_raw) / (self.ecg_length+5))
                ecg_raw = f(xnew)

            X_teacher = np.expand_dims(np.concatenate([eda_features, ppg_features, resp_features, eeg_features, ecg_features]),0)
            if self.transform:
                X_teacher = self.scaler.transform(X_teacher)
            ecg_raw = (ecg_raw[:self.ecg_length] - 2169.707228404167) / 329.4425863704116
            ecg_raw = ecg_raw.reshape((106, 106))
            X_student = np.expand_dims(ecg_raw, -1)

            if split:
                yield X_teacher.flatten(), X_student, self.convertBinLabel(val_label), self.convertBinLabel(ar_label)
            else:
                yield X_teacher.flatten(), X_student, self.convertLabel(val_label, ar_label)
            i += 1


    def convertBinLabel(self, label, th=3):
        labels = np.zeros_like(label)
        labels[label>= th] = 1
        return labels

    def convertLabel(self, val, ar, th=3):
        labels = np.zeros_like(val)
        labels[(val <= th) & (ar <= th)] = 0
        labels[(val <= th) & (ar > th)] = 1
        labels[(val > th) & (ar <= th)] = 2
        labels[(val > th) & (ar > th)] = 3


        return labels.astype(int)


