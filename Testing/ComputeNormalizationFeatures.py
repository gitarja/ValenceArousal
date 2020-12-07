from KnowledgeDistillation.Utils.DataGenerator import DataGenerator
import numpy as np
from Conf.Settings import DATASET_PATH
import glob
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import pandas as pd
from joblib import dump, load



features = []
y_ar = []
y_val = []

data_path = DATASET_PATH+"*"
game_result = "\\*_gameResults.csv"
path_result = "results\\"

for folder in glob.glob(data_path):
    for subject in glob.glob(folder + "\\*-2020-*"):
        eeg_path = subject + "\\results\\EEG\\"
        eda_path = subject + "\\results\\eda\\"
        ppg_path = subject + "\\results\\ppg\\"
        resp_path = subject + "\\results\\Resp\\"
        ecg_path = subject + "\\results\\ECG\\"
        ecg_resp_path = subject + "\\results\\ECG_resp\\"

        features_list = pd.read_csv(subject + "\\features_list.csv")
        for i in range(len(features_list)):
            filename = features_list.iloc[i]["Idx"]
            eda_features = np.load(eda_path + "eda_" + str(filename) + ".npy")
            ppg_features = np.load(ppg_path + "ppg_" + str(filename) + ".npy")
            resp_features = np.load(resp_path + "resp_" + str(filename) + ".npy")
            eeg_features = np.load(eeg_path + "eeg_" + str(filename) + ".npy")
            ecg_features = np.load(ecg_path + "ecg_" + str(filename) + ".npy")
            ecg_resp_features = np.load(ecg_resp_path + "ecg_resp_" + str(filename) + ".npy")

            concat_features = np.concatenate(
                [eda_features, ppg_features, resp_features, ecg_resp_features, ecg_features, eeg_features ])
            if np.sum(np.isinf(concat_features)) == 0 & np.sum(np.isnan(concat_features)) == 0:
                # print(eda_features.shape)
                features.append(concat_features)
                y_ar.append(features_list.iloc[i]["Arousal"])
                y_val.append(features_list.iloc[i]["Valence"])
            else:
                print(subject + "_" + str(i))

    # concatenate features and normalize them
X = np.concatenate([features])

#max
max_scaller = MaxAbsScaler().fit(X)
X_st = max_scaller.transform(X)


norm_scaller = StandardScaler().fit(X_st)
X_norm = norm_scaller.transform(X_st)

np.save('..\\KnowledgeDistillation\\Utils\\max.npy', np.max(X, axis=0))
np.save('..\\KnowledgeDistillation\\Utils\\mean.npy', np.mean(X, axis=0))
np.save('..\\KnowledgeDistillation\\Utils\\std.npy', np.std(X, axis=0))

