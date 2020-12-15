from ECG.ECGFeatures import ECGFeatures
import pandas as pd
from Libs.Utils import timeToInt
import numpy as np
from Conf.Settings import SPLIT_TIME, FS_ECG, STRIDE, EXTENTION_TIME, ECG_R_PATH, ECG_PATH, DATASET_PATH
from os import path
import os
import glob

ecg_file = "\\ECG\\"
game_result = "\\*_gameResults.csv"

for folder in glob.glob(DATASET_PATH + "*"):
    for subject in glob.glob(folder + "\\*-2020-10-*"):
        print(subject)
        try:
            os.makedirs(subject + ECG_PATH, exist_ok=True)
            data = pd.read_csv(glob.glob(subject + ecg_file + "*.csv")[0])
            data_EmotionTest = pd.read_csv(glob.glob(subject + game_result)[0])

            # convert timestamp to int
            data.loc[:, 'timestamp'] = data.loc[:, 'timestamp'].apply(timeToInt)
            data_EmotionTest.loc[:, 'Time_Start'] = data_EmotionTest.loc[:, 'Time_Start'].apply(timeToInt)
            data_EmotionTest.loc[:, 'Time_End'] = data_EmotionTest.loc[:, 'Time_End'].apply(timeToInt)

            # features extractor
            featuresExct = ECGFeatures(FS_ECG)
            emotionTestResult = pd.DataFrame(
                columns=["Idx", "Start", "End", "Valence", "Arousal", "Emotion", "Status", "Subject"])

            idx = 0
            for i in range(len(data_EmotionTest)):
                tdelta = data_EmotionTest.iloc[i]["Time_End"] - data_EmotionTest.iloc[i]["Time_Start"]
                time_end = data_EmotionTest.iloc[i]["Time_End"]
                valence = data_EmotionTest.iloc[i]["Valence"]
                arousal = data_EmotionTest.iloc[i]["Arousal"]
                emotion = data_EmotionTest.iloc[i]["Emotion"]

                for j in np.arange(0, (tdelta // SPLIT_TIME), STRIDE):
                    # take 2.5 sec after end
                    # end = time_end - ((j - 1) * SPLIT_TIME) + EXTENTION_TIME
                    # start = time_end - (j * SPLIT_TIME)

                    end = time_end - ((j) * SPLIT_TIME)
                    start = time_end - ((j + 1) * SPLIT_TIME) - EXTENTION_TIME

                    ecg = data[(data["timestamp"].values >= start) & (data["timestamp"].values <= end)]
                    status = 0

                    # extract ecg features
                    ecg_values = ecg['ecg'].values
                    time_domain = featuresExct.extractTimeDomain(ecg_values)
                    freq_domain = featuresExct.extractFrequencyDomain(ecg_values)
                    nonlinear_domain = featuresExct.extractNonLinearDomain(ecg_values)
                    if time_domain.shape[0] != 0 and freq_domain.shape[0] != 0 and nonlinear_domain.shape[0] != 0:
                        concatenate_features = np.concatenate([time_domain, freq_domain, nonlinear_domain])
                        if np.sum(np.isinf(concatenate_features)) == 0 & np.sum(np.isinf(concatenate_features)) == 0:
                            np.save(subject + ECG_PATH + "ecg_" + str(idx) + ".npy", concatenate_features)
                            # save raw ecg data
                            # np.save(subject + path_result_raw + "ecg_raw_" + str(idx) + ".npy", ecg['ecg'].values)
                            if not path.exists(subject + ECG_R_PATH):
                                os.mkdir(subject + ECG_R_PATH)
                            np.save(subject + ECG_R_PATH + "ecg_raw_" + str(idx) + ".npy", ecg_values)
                            status = 1
                        else:
                            status = 0

                    # add object to dataframes
                    subject_name = subject.split("\\")[-1]
                    emotionTestResult = emotionTestResult.append(
                        {"Idx": str(idx), "Subject": subject_name, "Start": str(start), "End": str(end),
                         "Valence": valence, "Arousal": arousal, "Emotion": emotion, "Status": status},
                        ignore_index=True)
                    idx += 1

            # save to csv
            emotionTestResult.to_csv(subject + "\\ECG_features_list_" + str(STRIDE) + ".csv", index=False)

        except:
            print("Error:" + subject)
