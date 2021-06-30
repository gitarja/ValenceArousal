import pandas as pd
import numpy as np
from Libs.Utils import timeToInt, dreamerLabelsConv, regressLabelsConv, emotionLabels, convertOneHotLabel
from Conf.Settings import ECG_PATH, RESP_PATH, EEG_PATH, ECG_RESP_PATH, EDA_PATH, PPG_PATH, DATASET_PATH, ECG_R_PATH, \
    ECG_RR_PATH, FS_ECG, ROAD_ECG, SPLIT_TIME, STRIDE, ECG_D_R_PATH, N_CLASS, DREAMER_PATH, N_SUBJECT, N_VIDEO_GENRE
from ECG.ECGFeatures import ECGFeatures
from joblib import Parallel, delayed


class DataFetch:

    def __init__(self, train_file=None, validation_file=None, test_file=None, ECG_N=None, KD=False, teacher=False,
                 ECG=False, EDA=False, PPG=False, RESP=False, training=True, high_only=False, multi_task=False):
        utils_path = "D:\\usr\\nishihara\\GitHub\\ValenceArousal\\Values\\"
        self.max = np.load(utils_path + "max.npy")
        self.mean = np.load(utils_path + "mean.npy")
        self.std = np.load(utils_path + "std.npy")
        self.mean_ecg = np.load(utils_path + "ECG\\ecg_mean.npy")
        self.std_ecg = np.load(utils_path + "ECG\\ecg_std.npy")
        self.mean_eda = np.load(utils_path + "EDA\\eda_mean.npy")
        self.std_eda = np.load(utils_path + "EDA\\eda_std.npy")
        self.mean_ppg = np.load(utils_path + "PPG\\ppg_mean.npy")
        self.std_ppg = np.load(utils_path + "PPG\\ppg_std.npy")
        self.mean_resp = np.load(utils_path + "RESP\\resp_mean.npy")
        self.std_resp = np.load(utils_path + "RESP\\resp_std.npy")

        self.KD = KD
        self.teacher = teacher
        self.ECG_N = ECG_N
        self.ECG = ECG
        self.EDA = EDA
        self.PPG = PPG
        self.RESP = RESP
        self.high_only = high_only
        self.multi_task = multi_task
        self.w = 0
        self.j = 0
        self.ecg_features = ECGFeatures(fs=FS_ECG)

        # normalization ecg features
        self.ecg_mean = np.array([2.18785670e+02, 5.34106162e+01, 1.22772848e+01, 8.87240641e+00,
                                  1.23045575e+01, 8.19622448e+00, 2.80084568e+02, 1.51193876e+01,
                                  3.36927105e+01, 6.63072895e+01, 7.52327656e-01, 1.85165308e+00,
                                  1.42787092e-01])

        self.ecg_std = np.array([28.6904681, 7.2190369, 8.96941273, 8.57895833, 13.34906982,
                                 10.67710367, 36.68525696, 9.31097392, 21.09139643, 21.09139643,
                                 0.88959446, 0.48770451, 0.08282199])
        self.ECG_N = ECG_N

        if training == True:
            self.data_train = self.readData(pd.read_csv(train_file), KD, True)
            self.train_n = len(self.data_train)

        self.data_test = self.readData(pd.read_csv(test_file), KD)
        self.data_val = self.readData(pd.read_csv(validation_file), KD)
        self.test_n = len(self.data_test)
        self.val_n = len(self.data_val)

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
            # data_set = self.data_test + self.data_val
            data_set = self.data_test
        i = 0

        # print(len(data_set))
        while i < len(data_set):
            # print(i)
            data_i = data_set[i]
            if self.teacher:
                if self.multi_task:
                    yield data_i[0], data_i[1], data_i[2], data_i[3], data_i[7], data_i[8], data_i[9]
                else:
                    yield data_i[0], data_i[1], data_i[2], data_i[3]
            else:
                if self.KD:
                    if self.ECG | self.EDA | self.PPG | self.RESP:
                        if self.multi_task:
                            yield data_i[0], data_i[1], data_i[2], data_i[3], data_i[5], data_i[6], data_i[7], data_i[8], data_i[9]
                        else:
                            yield data_i[0], data_i[1], data_i[2], data_i[3], data_i[5], data_i[6]
                    else:
                        if self.multi_task:
                            yield data_i[0], data_i[1], data_i[2], data_i[3], data_i[4], data_i[6], data_i[7], data_i[8], data_i[9]
                        else:
                            yield data_i[0], data_i[1], data_i[2], data_i[3], data_i[4], data_i[6]
                else:
                    yield data_i[1], data_i[2], data_i[3], data_i[4]

            i += 1
        self.j += 1

    def readData(self, features_list, KD, training=False):
        data_set = []
        # features_list = features_list.sample(frac=1.)

        # if training:
        #     val_features_neg = features_list[(features_list["Valence_convert"] < 0)].sample(frac=1.1, replace=True)
        #     val_features_pos = features_list[features_list["Valence_convert"] >= 0].sample(frac=0.9)
        #     features_list = pd.concat([val_features_neg, val_features_pos])
        for i in range(len(features_list)):
            filename = features_list.iloc[i]["Idx"]
            base_path = DATASET_PATH + features_list.iloc[i]["Subject"][3:] + "\\" + features_list.iloc[i][
                "Subject"]
            eda_features_file = base_path + EDA_PATH + "eda_" + str(filename) + ".npy"
            ppg_features_file = base_path + PPG_PATH + "ppg_" + str(filename) + ".npy"
            resp_features_file = base_path + RESP_PATH + "resp_" + str(filename) + ".npy"
            eeg_features_file = base_path + EEG_PATH + "eeg_" + str(filename) + ".npy"
            ecg_features_file = base_path + ECG_PATH + "ecg_" + str(filename) + ".npy"
            ecg_resp_features_file = base_path + ECG_RESP_PATH + "ecg_resp_" + str(filename) + ".npy"
            ecg_raw_file = base_path + ECG_R_PATH + "ecg_raw_" + str(filename) + ".npy"
            # if KD:
            files = [eda_features_file, ppg_features_file, resp_features_file, ecg_resp_features_file,
                     ecg_features_file, eeg_features_file, ecg_raw_file]
            features = Parallel(n_jobs=7)(delayed(np.load)(files[j]) for j in range(len(files)))
            ecg = features[6]
            features_student = []
            ecg_features = features[4]
            eda_features = features[0]
            ppg_features = features[1]
            resp_features = features[2]
            if self.ECG:
                ecg_features = (ecg_features - self.mean_ecg) / self.std_ecg
                features_student.append(ecg_features)
            if self.EDA:
                eda_features = (eda_features - self.mean_eda) / self.std_eda
                features_student.append(eda_features)
            if self.PPG:
                ppg_features = (ppg_features - self.mean_ppg) / self.std_ppg
                features_student.append(ppg_features)
            if self.RESP:
                resp_features = (resp_features - self.mean_resp) / self.std_resp
                features_student.append(resp_features)
            if features_student:
                features_student = np.concatenate(features_student)

            # else:
            #     files = [eda_features_file, ppg_features_file, resp_features_file, ecg_resp_features_file, ecg_features_file, eeg_features_file
            #              ]
            #     features = Parallel(n_jobs=6)(delayed(np.load)(files[j]) for j in range(len(files)))

            concat_features = np.concatenate(features[0:6])

            # if np.sum(np.isinf(concat_features)) == 0 & np.sum(np.isnan(concat_features)) == 0:
            concat_features_norm = (concat_features - self.mean) / self.std

            # print(np.max(concat_features_norm))
            # print(np.min(concat_features[575:588]))
            y_ar = features_list.iloc[i]["Arousal"]
            y_val = features_list.iloc[i]["Valence"]
            emotions = features_list.iloc[i]["Emotion"]
            subject = features_list.iloc[i]["Subject_label"]
            gender = features_list.iloc[i]["Gender_label"]
            video = features_list.iloc[i]["Video_genre"]

            # convert the label either to binary class or three class

            # y_ar_bin = arToLabels(y_ar) #binary labels of arousal
            # y_val_bin = valToLabels(y_val) #binary labels of valence
            # ar_weight = arWeight(y_ar_bin) #weight for arousal samples
            # val_weight = valWeight(y_val_bin) #valence for arousal samples

            y_emotions = emotionLabels(emotions, N_CLASS)
            y_r_ar = regressLabelsConv(y_ar)
            y_r_val = regressLabelsConv(y_val)
            y_subject = convertOneHotLabel(subject, N_SUBJECT)
            y_video = convertOneHotLabel(video, N_VIDEO_GENRE)

            # if len(ecg) >= self.ECG_N:
            #     ecg = (ecg - 2140.397356669409) / 370.95493558685325
            #     ecg = ecg[-self.ECG_N:]
            #     # label = np.zeros_like(ecg[-self.ECG_N:]) - 1
            #     # label[self.ecg_features.extractRR(ecg).astype(np.int32)] = 1.
            #     # ecg_features = (features[4] - self.ecg_mean) / self.ecg_std
            #     # w = features_list.iloc[i]["weight"]
            #     # if w == 0.5:
            #     #     w += 0.3
            #     w = 1
            #
            #     # resampling data
            #     # if training:
            #     #     if (y_r_ar < 0 and y_val < 0) or (y_r_ar> 0 and y_val< 0):
            #     #         w = 2.
            #     if self.high_only:
            #         if w == 1:
            #             data_set.append([concat_features_norm, y_emotions, y_r_ar, y_r_val, ecg, features_student, w])
            #     else:
            #         data_set.append([concat_features_norm, y_emotions, y_r_ar, y_r_val, ecg, features_student, w])

            ecg = (ecg - 2140.397356669409) / 370.95493558685325
            ecg = ecg[-self.ECG_N:]
            w = 1
            if self.high_only:
                if w == 1:
                    data_set.append([concat_features_norm, y_emotions, y_r_ar, y_r_val, ecg, features_student, w, y_subject, gender, y_video])
            else:
                data_set.append([concat_features_norm, y_emotions, y_r_ar, y_r_val, ecg, features_student, w, y_subject, gender, y_video])

        return data_set

    def randomECG(self, ecg):
        diff_n = len(ecg) - self.ECG_N
        start = np.random.randint(1, diff_n, size=1)[0]
        end = start + self.ECG_N

        return ecg[start:end]

