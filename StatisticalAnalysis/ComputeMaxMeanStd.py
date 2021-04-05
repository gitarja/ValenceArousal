import numpy as np
import pandas as pd
import glob
import os
from joblib import Parallel, delayed
from Conf.Settings import DATASET_PATH, EDA_PATH, PPG_PATH, RESP_PATH, EEG_PATH, ECG_PATH, ECG_RESP_PATH, STRIDE


all_features = []
eeg_features = []
ecg_features = []
eda_features = []
ppg_features = []
resp_features = []
for folder in glob.glob(DATASET_PATH + "2020-*\\"):
    for subject in glob.glob(folder + "*-2020-*\\"):
        features_list = pd.read_csv(subject + "features_list_" + str(STRIDE) + ".csv")

        for i in range(len(features_list)):
            filename = features_list.iloc[i]["Idx"]
            eda_features_file = subject + EDA_PATH + "eda_" + str(filename) + ".npy"
            ppg_features_file = subject + PPG_PATH + "ppg_" + str(filename) + ".npy"
            resp_features_file = subject + RESP_PATH + "resp_" + str(filename) + ".npy"
            eeg_features_file = subject + EEG_PATH + "eeg_" + str(filename) + ".npy"
            ecg_features_file = subject + ECG_PATH + "ecg_" + str(filename) + ".npy"
            ecg_resp_features_file = subject + ECG_RESP_PATH + "ecg_resp_" + str(filename) + ".npy"
            files = [eda_features_file, ppg_features_file, resp_features_file, ecg_resp_features_file, eeg_features_file, ecg_features_file]
            concat_features = Parallel(n_jobs=6)(delayed(np.load)(f) for f in files)
            # eeg_features.append(concat_features[-2])
            ecg_features.append(concat_features[-1])
            eda_features.append(concat_features[0])
            ppg_features.append(concat_features[1])
            resp_features.append(concat_features[2])
            all_features.append(np.concatenate(concat_features))

# eeg_features = np.concatenate(eeg_features)
ecg_features = np.concatenate(ecg_features)
eda_features = np.concatenate(eda_features)
ppg_features = np.concatenate(ppg_features)
resp_features = np.concatenate(resp_features)
all_features = np.concatenate(all_features)

max_value = all_features.max(axis=0)
min_value = all_features.min(axis=0)
mean_value = all_features.mean(axis=0)
std_value = all_features.std(axis=0)
# max_value_eeg = eeg_features.max(axis=0)
# min_value_eeg = eeg_features.min(axis=0)
# mean_value_eeg = eeg_features.mean(axis=0)
# std_value_eeg = eeg_features.std(axis=0)
max_value_ecg = ecg_features.max(axis=0)
min_value_ecg = ecg_features.min(axis=0)
mean_value_ecg = ecg_features.mean(axis=0)
std_value_ecg = ecg_features.std(axis=0)
max_value_eda = eda_features.max(axis=0)
min_value_eda = eda_features.min(axis=0)
mean_value_eda = eda_features.mean(axis=0)
std_value_eda = eda_features.std(axis=0)
max_value_ppg = ppg_features.max(axis=0)
min_value_ppg = ppg_features.min(axis=0)
mean_value_ppg = ppg_features.mean(axis=0)
std_value_ppg = ppg_features.std(axis=0)
max_value_resp = resp_features.max(axis=0)
min_value_resp = resp_features.min(axis=0)
mean_value_resp = resp_features.mean(axis=0)
std_value_resp = resp_features.std(axis=0)

path_save = "D:\\usr\\nishihara\\GitHub\\ValenceArousal\\Values\\"
os.makedirs(path_save + "ECG\\", exist_ok=True)
os.makedirs(path_save + "EDA\\", exist_ok=True)
os.makedirs(path_save + "PPG\\", exist_ok=True)
os.makedirs(path_save + "RESP\\", exist_ok=True)
# os.makedirs(path_save + "EEG\\", exist_ok=True)
np.save(path_save + "max.npy", max_value)
np.save(path_save + "min.npy", min_value)
np.save(path_save + "mean.npy", mean_value)
np.save(path_save + "std.npy", std_value)
# np.save(path_save + "eeg_max.npy", max_value_eeg)
# np.save(path_save + "eeg_min.npy", min_value_eeg)
# np.save(path_save + "eeg_mean.npy", mean_value_eeg)
# np.save(path_save + "eeg_std.npy", std_value_eeg)
np.save(path_save + "ECG\\ecg_max.npy", max_value_ecg)
np.save(path_save + "ECG\\ecg_min.npy", min_value_ecg)
np.save(path_save + "ECG\\ecg_mean.npy", mean_value_ecg)
np.save(path_save + "ECG\\ecg_std.npy", std_value_ecg)
np.save(path_save + "EDA\\eda_max.npy", max_value_eda)
np.save(path_save + "EDA\\eda_min.npy", min_value_eda)
np.save(path_save + "EDA\\eda_mean.npy", mean_value_eda)
np.save(path_save + "EDA\\eda_std.npy", std_value_eda)
np.save(path_save + "PPG\\ppg_max.npy", max_value_ppg)
np.save(path_save + "PPG\\ppg_min.npy", min_value_ppg)
np.save(path_save + "PPG\\ppg_mean.npy", mean_value_ppg)
np.save(path_save + "PPG\\ppg_std.npy", std_value_ppg)
np.save(path_save + "RESP\\resp_max.npy", max_value_resp)
np.save(path_save + "RESP\\resp_min.npy", min_value_resp)
np.save(path_save + "RESP\\resp_mean.npy", mean_value_resp)
np.save(path_save + "RESP\\resp_std.npy", std_value_resp)

