import pandas as pd
import os
import numpy as np
import random
from scipy import signal
from Libs.Utils import valArLevelToLabels
from Conf.Settings import ECG_PATH, RESP_PATH, EEG_PATH, ECG_RESP_PATH, EDA_PATH, PPG_PATH, DATASET_PATH, ECG_RAW_PATH

from joblib import load


class DataFetch:

    def __init__(self, train_file, test_file, validation_file, ECG_N, max_scaler=None, norm_scaler=None):
        self.data_train = pd.read_csv(train_file)
        self.data_test = pd.read_csv(test_file)
        self.data_val = pd.read_csv(validation_file)
        self.ECG_N = ECG_N

        self.train_n = len(self.data_train)
        self.val_n = len(self.data_val)
        self.test_n = len(self.data_test)

        # self.max = np.load("Utils\\max.npy")
        # self.mean = np.load("Utils\\mean.npy")
        # self.std = np.load("Utils\\std.npy")

    def fetch(self, training_mode=0, KD=False):
        '''

        :param training_mode: 0 = training, 1 = testing, 2 = validation
        :return:
        '''
        if training_mode == 0:
            features_list = self.data_train
        elif training_mode == 1:
            features_list = self.data_val
        else:
            features_list = self.data_test
        i = 0
        while i < len(features_list):
                filename = features_list.iloc[i]["Idx"]
                base_path = DATASET_PATH + features_list.iloc[i]["Subject"][3:] + "\\" + features_list.iloc[i][
                    "Subject"]
                eda_features = np.load(base_path + EDA_PATH + "eda_" + str(filename) + ".npy")
                ppg_features = np.load(base_path + PPG_PATH + "ppg_" + str(filename) + ".npy")
                resp_features = np.load(base_path + RESP_PATH + "resp_" + str(filename) + ".npy")
                eeg_features = np.load(base_path + EEG_PATH + "eeg_" + str(filename) + ".npy")
                ecg_features = np.load(base_path + ECG_PATH + "ecg_" + str(filename) + ".npy")
                ecg_resp_features = np.load(base_path + ECG_RESP_PATH + "ecg_resp_" + str(filename) + ".npy")

                concat_features = np.concatenate(
                    [eda_features, ppg_features, resp_features, ecg_resp_features, ecg_features, eeg_features])

                if np.sum(np.isinf(concat_features)) == 0 & np.sum(np.isnan(concat_features)) == 0:
                    # concat_features_norm = (concat_features - self.mean) / self.std
                    # print(np.max(concat_features_norm))
                    # print(np.min(concat_features[575:588]))
                    y_ar = valArLevelToLabels(features_list.iloc[i]["Arousal"])
                    y_val = valArLevelToLabels(features_list.iloc[i]["Valence"])

                    if KD == False:
                        return np.array([concat_features[:1102], concat_features[1102:1150], concat_features[1150:]]), y_ar, y_val
                    else:
                        ecg = np.load(base_path + ECG_RAW_PATH + "ecg_raw_" + str(filename) + ".npy")
                        return concat_features, y_ar, y_val, ecg[:self.ECG_N]
                i+=1
