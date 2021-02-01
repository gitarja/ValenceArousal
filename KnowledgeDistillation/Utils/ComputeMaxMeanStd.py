import numpy as np
import pandas as pd
import glob
from joblib import Parallel, delayed
from Conf.Settings import DATASET_PATH, EDA_PATH, PPG_PATH, RESP_PATH, EEG_PATH, ECG_PATH, ECG_RESP_PATH, STRIDE

def loadFeature(file, idx):
    feature = np.load(file)
    return feature, idx

features = []
for folder in glob.glob(DATASET_PATH + "2020-*\\"):
    for subject in glob.glob(folder + "*-2020-*\\"):
        features_list = pd.read_csv(subject + "features_list_" + str(STRIDE) + ".csv")

        for i in range(len(features_list)):
            filename = features_list.iloc[i]["Idx"]
            eda_features = subject + EDA_PATH + "eda_" + str(filename) + ".npy"
            ppg_features = subject + PPG_PATH + "ppg_" + str(filename) + ".npy"
            resp_features = subject + RESP_PATH + "resp_" + str(filename) + ".npy"
            eeg_features = subject + EEG_PATH + "eeg_" + str(filename) + ".npy"
            ecg_features = subject + ECG_PATH + "ecg_" + str(filename) + ".npy"
            ecg_resp_features = subject + ECG_RESP_PATH + "ecg_resp_" + str(filename) + ".npy"
            files = [eda_features, ppg_features, resp_features, ecg_resp_features, ecg_features, eeg_features]
            features = Parallel(n_jobs=6)(delayed(np.load)(f) for f in enumerate(files))



