from ECG.ECGFeatures import ECGFeatures
import pandas as pd
from Libs.Utils import timeToInt, arToLabels
import numpy as np
from Conf.Settings import SPLIT_TIME, STRIDE, EXTENTION_TIME, ECG_R_PATH, ECG_PATH, DATASET_PATH, RESULTS_PATH, \
    DREAMER_PATH, DREAMER_FS_ECG, NUM_SUBJECTS, NUM_VIDEO_SEQ
from os import path
import os
import glob

ecg_file = "\\ECG\\"
game_result = "\\*_gameResults.csv"

data_label = pd.read_csv(DREAMER_PATH + "labels.csv")
for subject in range(1, NUM_SUBJECTS + 1):
    print("subject_" + str(subject))
    subject_path = DREAMER_PATH + "subject_" + str(subject)
    if not path.exists(subject_path + RESULTS_PATH):
        os.mkdir(subject_path + RESULTS_PATH)

    os.makedirs(subject_path + ECG_PATH, exist_ok=True)
    idx = 0
    features_list = pd.DataFrame(columns=["Idx", "Valence", "Arousal", "Status", "Subject"])
    for i, video_seq in enumerate(range(1, NUM_VIDEO_SEQ + 1)):
        file = subject_path + ecg_file + "filtered_ecg_" + str(subject) + "_" + str(video_seq) + ".csv"
        data = pd.read_csv(file)

        timestamp = pd.DataFrame(np.linspace(0, len(data) / DREAMER_FS_ECG, len(data)), columns=["timestamp"])
        data = pd.concat([timestamp, data], axis=1)

        # features extractor
        featuresExct = ECGFeatures(DREAMER_FS_ECG)
        arousal = \
        data_label[(data_label["Subject"] == subject) & (data_label["VideoSequence"] == video_seq)]["Arousal"].values[0]
        valence = \
        data_label[(data_label["Subject"] == subject) & (data_label["VideoSequence"] == video_seq)]["Valence"].values[0]
        time_end = data.iloc[-1]["timestamp"]

        for j in np.arange(0, (time_end // SPLIT_TIME), STRIDE):
            # take 2.5 sec after end
            # end = time_end - ((j - 1) * SPLIT_TIME) + EXTENTION_TIME
            # start = time_end - (j * SPLIT_TIME)

            end = time_end - (j * SPLIT_TIME)
            start = time_end - ((j + 1) * SPLIT_TIME) - EXTENTION_TIME

            ecg = data[(data["timestamp"].values >= start) & (data["timestamp"].values <= end)]
            status = 0

            # extract ecg features
            ecg_values_ch1 = ecg['CH1'].values
            ecg_values_ch2 = ecg['CH2'].values
            time_domain = np.concatenate(
                [featuresExct.extractTimeDomain(ecg_val) for ecg_val in [ecg_values_ch1, ecg_values_ch2]])
            freq_domain = np.concatenate(
                [featuresExct.extractFrequencyDomain(ecg_val) for ecg_val in [ecg_values_ch1, ecg_values_ch2]])
            nonlinear_domain = np.concatenate(
                [featuresExct.extractNonLinearDomain(ecg_val) for ecg_val in [ecg_values_ch1, ecg_values_ch2]])
            if time_domain.shape[0] != 0 and freq_domain.shape[0] != 0 and nonlinear_domain.shape[0] != 0:
                concatenate_features = np.concatenate([time_domain, freq_domain, nonlinear_domain])
                if np.sum(np.isinf(concatenate_features)) == 0 & np.sum(np.isinf(concatenate_features)) == 0:
                    if not path.exists(subject_path + ECG_PATH):
                        os.mkdir(subject_path + ECG_PATH)
                    np.save(subject_path + ECG_PATH + "ecg_" + str(idx) + ".npy", concatenate_features)
                    # save raw ecg data
                    # np.save(subject + path_result_raw + "ecg_raw_" + str(idx) + ".npy", ecg['ecg'].values)
                    if not path.exists(subject_path + ECG_R_PATH):
                        os.mkdir(subject_path + ECG_R_PATH)
                    np.save(subject_path + ECG_R_PATH + "ecg_raw_" + str(idx) + ".npy", ecg_values_ch1)
                    status = 1
                else:
                    status = 0

            # add object to dataframes
            features_list = features_list.append(
                {"Idx": idx, "Subject": subject, "Valence": valence, "Arousal": arousal, "Status": status},
                ignore_index=True)
            idx += 1

    # save to csv
    features_list.to_csv(subject_path + "\\ECG_features_list_" + str(STRIDE) + ".csv", index=False)

    # try:
    #
    # except:
    #     print("Error: subject_" + str(subject))


