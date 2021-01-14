import pandas as pd
import os
import numpy as np
import random
from scipy import signal
from Libs.Utils import valToLabels, arToLabels, arValMulLabels, valArToLabels
from Conf.Settings import ECG_PATH, RESP_PATH, EEG_PATH, ECG_RESP_PATH, EDA_PATH, PPG_PATH, DATASET_PATH, ECG_R_PATH, ECG_RR_PATH, FS_ECG
from ECG.ECGFeatures import ECGFeatures
from joblib import Parallel, delayed


class DataFetch:

    def __init__(self, train_file, validation_file, test_file, ECG_N, KD=False, multiple=False, soft=False):
        self.max = np.load("Utils\\max.npy")
        self.mean = np.load("Utils\\mean.npy")
        self.std = np.load("Utils\\std.npy")

        self.KD = KD
        self.multiple = multiple
        self.ECG_N = ECG_N
        self.soft = soft

        self.data_train = self.readData(pd.read_csv(train_file), KD, True)
        self.data_val = self.readData(pd.read_csv(validation_file), KD)
        self.data_test = self.readData(pd.read_csv(test_file), KD)


        self.ECG_N = ECG_N

        self.train_n = len(self.data_train)
        self.val_n = len(self.data_val)
        self.test_n = len(self.data_test)



    def fetch(self, training_mode=0):
        '''

        :param training_mode: 0 = training, 1 = testing, 2 = validation
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
            if self.multiple:
                if self.KD:
                    # print(len(data_i))
                    yield data_i[0], data_i[1], data_i[2], data_i[3], data_i[4]
                else:
                    yield data_i[0], data_i[1], data_i[2], data_i[3]
            else:
                if self.KD:
                    # print(len(data_i))
                    yield data_i[0], data_i[1], data_i[2],  data_i[4]
                else:
                    yield data_i[0], data_i[1], data_i[2],
            i += 1


    def readData(self, features_list, KD, training=False):
        data_set = []
        for i in range(len(features_list)):
            filename = features_list.iloc[i]["Idx"]
            base_path = DATASET_PATH + features_list.iloc[i]["Subject"][3:] + "\\" + features_list.iloc[i][
                "Subject"]
            eda_features = base_path + EDA_PATH + "eda_" + str(filename) + ".npy"
            ppg_features = base_path + PPG_PATH + "ppg_" + str(filename) + ".npy"
            resp_features = base_path + RESP_PATH + "resp_" + str(filename) + ".npy"
            eeg_features = base_path + EEG_PATH + "eeg_" + str(filename) + ".npy"
            ecg_features = base_path + ECG_PATH + "ecg_" + str(filename) + ".npy"
            ecg_resp_features = base_path + ECG_RESP_PATH + "ecg_resp_" + str(filename) + ".npy"
            ecg_raw = base_path + ECG_R_PATH + "ecg_raw_" + str(filename) + ".npy"
            if KD:
                files = [eda_features, ppg_features, resp_features, ecg_resp_features, ecg_features, eeg_features, ecg_raw]
                features = Parallel(n_jobs=7)(delayed(np.load)(files[j]) for j in range(len(files)))
                ecg = features[6]
            else:
                files = [eda_features, ppg_features, resp_features, ecg_resp_features, ecg_features, eeg_features
                         ]
                features = Parallel(n_jobs=6)(delayed(np.load)(files[j]) for j in range(len(files)))

            concat_features = np.concatenate(features[0:6])

            # if np.sum(np.isinf(concat_features)) == 0 & np.sum(np.isnan(concat_features)) == 0:
            concat_features_norm = (concat_features - self.mean) / self.std

            # print(np.max(concat_features_norm))
            # print(np.min(concat_features[575:588]))
            y_ar = features_list.iloc[i]["Arousal"]
            y_val = features_list.iloc[i]["Valence"]

            #convert the label either to binary class or three class
            if self.soft is False:
                y_ar_bin = arToLabels(y_ar)
                y_val_bin = valToLabels(y_val)
            else:
                y_ar_bin = valArToLabels(y_ar, True)
                y_val_bin = valArToLabels(y_val, True)

            m_class = arValMulLabels(y_ar_bin, y_val_bin)
            if KD :
                if len(ecg) >= self.ECG_N:
                    # ecg = (ecg - 2.7544520692684414e-06) / 0.15695187777333394
                    # ecg = (ecg -  1223.901793051745) / 1068.7720750244841
                    ecg = ecg / (4095 - 0)
                    if training:
                        ecg = self.randomECG(ecg)
                    else:
                        ecg = ecg[-self.ECG_N:]
                    # ecg = ecg /  2.0861534577149707
                    data_set.append([concat_features_norm, y_ar_bin, y_val_bin, m_class,  ecg])
            else:
                # data_set.append([concat_features_norm, y_ar_bin, y_val_bin, m_class])
                data_set.append([concat_features[-1343:-1330], y_ar_bin, y_val_bin, m_class])

        return data_set

    def randomECG(self, ecg):
        diff_n = len(ecg) - self.ECG_N
        start = np.random.randint(1, diff_n, size=1)[0]
        end = start + self.ECG_N

        return ecg[start:end]



class DataFetchPreTrain:

    def __init__(self, train_file, validation_file, test_file, ECG_N):

        self.ECG_N = ECG_N
        self.ecg_features = ECGFeatures(fs=FS_ECG)
        self.data_train = self.readData(pd.read_csv(train_file))
        self.data_val = self.readData(pd.read_csv(validation_file))
        self.data_test = self.readData(pd.read_csv(test_file))


        self.ECG_N = ECG_N

        self.train_n = len(self.data_train)
        self.val_n = len(self.data_val)
        self.test_n = len(self.data_test)



    def fetch(self, training_mode=0):
        '''

        :param training_mode: 0 = training, 1 = testing, 2 = validation
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
            yield data_i[0], data_i[1]
            i += 1


    def readData(self, features_list):
        data_set = []
        for i in range(len(features_list)):
            filename = features_list.iloc[i]["Idx"]
            base_path = DATASET_PATH + features_list.iloc[i]["Subject"][3:] + "\\" + features_list.iloc[i][
                "Subject"]

            # ecg_features = base_path + ECG_PATH + "ecg_" + str(filename) + ".npy"
            ecg_raw = base_path + ECG_R_PATH + "ecg_raw_" + str(filename) + ".npy"

            files = [ecg_raw]
            features = Parallel(n_jobs=2)(delayed(np.load)(files[j]) for j in range(len(files)))
            ecg = features[-1]

            # concat_features = features[0]


            if len(ecg) >= self.ECG_N:

                    ecg = ecg[-self.ECG_N:] / (4095 - 0)
                    label = np.zeros_like(ecg[-self.ECG_N:])
                    label[self.ecg_features.extractRR(ecg).astype(np.int32)] = 1
                    data_set.append([ecg[-self.ECG_N:], label])
                    # data_set.append([ecg[-self.ECG_N:], concat_features[1]])

        return data_set

