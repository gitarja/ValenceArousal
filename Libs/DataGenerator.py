import pandas as pd
import numpy as np
class DataGenerator():

    def __init__(self, teacher_features_path, student_features_path, features_list_file, val_th=3, ar_th=3, ecg_length=10000):
        '''
        :param teacher_features_path: directory path for teacher's features
        :param student_features_path: directory path for student's features
        :param features_list_path: file about the list of the features
        :param val_th: threshold deciding the valence valence
        :param ar_th: threshold deciding the arousal valence
        '''
        self.teacher_features_path = teacher_features_path
        self.student_features_path = student_features_path

        self.eeg_path = self.teacher_features_path + "EEG\\"
        self.gsr_path = self.teacher_features_path + "GSR\\"
        self.resp_path = self.teacher_features_path + "Resp\\"
        self.ecg_path = self.teacher_features_path + "ECG\\"

        self.ecg_raw_path = self.student_features_path + "ECG_raw\\"
        self.ecg_length = ecg_length

        self.features_list = pd.read_csv(features_list_file)



    def fetchData(self):

        for index, row in self.features_list.iterrows():
            filename = row["Idx"]
            val_label = row["Valence"]
            ar_label = row["Arousal"]
            eda_features = np.load(self.gsr_path + "eda_" + str(filename) + ".npy")
            ppg_features = np.load(self.gsr_path + "ppg_" + str(filename) + ".npy")
            resp_features = np.load(self.resp_path + "resp_" + str(filename) + ".npy")
            eeg_features = np.load(self.eeg_path + "eeg_" + str(filename) + ".npy")
            ecg_features = np.load(self.ecg_path + "ecg_" + str(filename) + ".npy")

            ecg_raw = np.load(self.ecg_raw_path + "ecg_raw_" + str(filename) + ".npy")

            X_teacher = np.expand_dims(np.concatenate([eda_features, ppg_features, resp_features, eeg_features, ecg_features]), 0)
            X_student = np.expand_dims(ecg_raw[:self.ecg_length], 0)

            return X_teacher, X_student, val_label, ar_label


    def convertLabel(self, val, ar):
        labels = np.zeros_like(val)
        labels[(val < 2.5) & (ar < 2.5)] = 1
        labels[(val < 2.5) & (ar > 2.5)] = 2
        labels[(val > 2.5) & (ar < 2.5)] = 3
        labels[(val > 2.5) & (ar > 2.5)] = 4

        return labels.astype(int)


