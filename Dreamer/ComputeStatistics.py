import numpy as np
import pandas as pd
import glob
import os
from joblib import Parallel, delayed
from Conf.Settings import DREAMER_PATH, EDA_PATH, PPG_PATH, RESP_PATH, EEG_PATH, ECG_PATH, ECG_RESP_PATH, STRIDE

features = []
for subject in glob.glob(DREAMER_PATH + "subject_*"):
    features_list = pd.read_csv(subject + "\\ECG_features_list_" + str(STRIDE) + ".csv")

    for i in range(len(features_list)):
        if features_list.iloc[i]["Status"] == 1:
            filename = features_list.iloc[i]["Idx"]
            ecg_features_file = subject + ECG_PATH + "ecg_" + str(filename) + ".npy"

            # files = [eda_features_file, ppg_features_file, resp_features_file, ecg_resp_features_file, ecg_features_file,
            #          eeg_features_file]
            # concat_features = Parallel(n_jobs=6)(delayed(np.load)(f) for f in files)
            files = ecg_features_file
            concat_features = np.load(files)
            # features.append(np.concatenate(concat_features))
            features.append(concat_features)

features = np.array(features)

max_value = features.max(axis=0)
min_value = features.min(axis=0)
mean_value = features.mean(axis=0)
std_value = features.std(axis=0)

path_save = "G:\\usr\\nishihara\\GitHub\\EmotionRecognition\\Dreamer\\Values\\"
os.makedirs(path_save, exist_ok=True)
np.save(path_save + "max.npy", max_value)
np.save(path_save + "min.npy", min_value)
np.save(path_save + "mean.npy", mean_value)
np.save(path_save + "std.npy", std_value)
