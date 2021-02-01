import pandas as pd
import numpy as np
from Conf.Settings import DATASET_PATH, ECG_R_PATH, EEG_R_PATH, EEG_RAW_MAX, EEG_RAW_MIN


class DataFetchPreTrain_CL:

    def __init__(self, train_file, validation_file, test_file, ECG_N, soft=False):

        self.ECG_N = ECG_N
        self.soft = soft
        self.data_train = self.readData(pd.read_csv(train_file))
        self.data_val = self.readData(pd.read_csv(validation_file))
        self.data_test = self.readData(pd.read_csv(test_file))
        self.train_n = len(self.data_train)
        self.val_n = len(self.data_val)
        self.test_n = len(self.data_test)

    def fetch(self, training_mode=0, ecg_or_eeg=0):
        '''
        :param training_mode: 0 = training, 1 = testing, 2 = validation
        :param ecg_or_eeg: 0 = ecg, 1 = eeg
        :return:
        '''
        if training_mode == 0:
            data_set = self.data_train
        elif training_mode == 1:
            data_set = self.data_val
        else:
            data_set = self.data_test
        i = 0
        # print(len(data_set))
        while i < len(data_set):
            # print(i)
            data_i = data_set[i]
            if ecg_or_eeg == 0:
                yield data_i[0], data_i[2], data_i[3]
            else:
                yield data_i[1], data_i[2], data_i[3]
            i += 1

    def readData(self, features_list):
        data_set = []
        for i in range(len(features_list)):
            filename = features_list.iloc[i]["Idx"]
            base_path = DATASET_PATH + features_list.iloc[i]["Subject"][3:] + "\\" + features_list.iloc[i][
                "Subject"]

            # ecg_features = base_path + ECG_PATH + "ecg_" + str(filename) + ".npy"
            ecg_raw = np.load(base_path + ECG_R_PATH + "ecg_raw_" + str(filename) + ".npy")
            eeg_raw = np.load(base_path + EEG_R_PATH + "eeg_raw_" + str(filename) + ".npy")

            time = features_list.iloc[i]["Start"]
            subject = features_list.iloc[i]["Subject"]

            if len(ecg_raw) >= self.ECG_N:
                ecg = (ecg_raw[-self.ECG_N:] - 1223.901793051745) / 1068.7720750244841
                eeg = (eeg_raw - EEG_RAW_MIN) / (EEG_RAW_MAX - EEG_RAW_MIN)
                data_set.append([ecg[-self.ECG_N:], eeg, time, subject])

        return data_set
