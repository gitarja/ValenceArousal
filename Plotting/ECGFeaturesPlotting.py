import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from Conf.Settings import DATASET_PATH, RESULTS_PATH, ECG_PATH, STRIDE, ECG_N, FS_ECG

FEATURES_NAME = ["RRI Mean", "RRI Counter", "RRI SD", "RRI Diff Mean", "RMSSD", "SDSD", "HR Mean", "HR Std"]

for date in glob.glob(DATASET_PATH + "2020-*\\"):
    for subject in glob.glob(date + "*-2020-*"):
        try:
            print(subject)
            path_plot = subject + "\\plot_ecg_features\\"
            os.makedirs(path_plot, exist_ok=True)
            features_list = pd.read_csv(subject + "\\features_list_" + str(STRIDE) + ".csv")
            # game_result = pd.read_csv(glob.glob(subject + "*_gameResults.csv")[0])

            video_seq = 0
            ecg_features_plotting = np.empty((0, ECG_N))
            for i, idx in enumerate(features_list["Idx"].values):
                arousal = features_list.iloc[i]["Arousal"]
                valence = features_list.iloc[i]["Valence"]
                if (i + 1) < len(features_list):
                    ar_delta = features_list.iloc[i + 1]["Arousal"] - features_list.iloc[i]["Arousal"]
                    val_delta = features_list.iloc[i + 1]["Valence"] - features_list.iloc[i]["Valence"]
                    t_delta = features_list.iloc[i + 1]["Start"] - features_list.iloc[i]["Start"]

                features = np.load(subject + ECG_PATH + "ecg_" + str(idx) + ".npy")
                features[0] *= 1000 / FS_ECG
                features[2] *= 1000 / FS_ECG
                features[6] /= 1000 / FS_ECG
                features[7] /= 1000 / FS_ECG

                features = np.expand_dims(features, axis=0)
                ecg_features_plotting = np.concatenate([ecg_features_plotting, features], axis=0)

                if ((t_delta >= 0) or ((ar_delta != 0) or (val_delta != 0))) or ((i + 1) >= len(features_list)):
                    fig = plt.figure()
                    for j, name in enumerate(FEATURES_NAME):
                        plt.subplot(4, 2, j + 1)
                        plt.plot(ecg_features_plotting[:, j])
                        plt.ylabel(name)
                    plt.tight_layout()
                    fig.savefig(path_plot + str(video_seq) + "_" + str(valence) + "_" + str(arousal) + ".png")
                    plt.close()
                    video_seq += 1
                    ecg_features_plotting = np.empty((0, ECG_N))

        except:
            print("Error:", subject)

