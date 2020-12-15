from GSR.GSRFeatures import PPGFeatures, EDAFeatures
import pandas as pd
from datetime import datetime
from Libs.Utils import timeToInt
from Conf.Settings import FS_GSR, SPLIT_TIME, STRIDE, EXTENTION_TIME, EDA_RAW_PATH, PPG_RAW_PATH, EDA_PATH, PPG_PATH, \
    DATASET_PATH
import numpy as np
import glob
import os

gsr_file = "\\GSR\\"
game_result = "\\*_gameResults.csv"

min_len = FS_GSR * (SPLIT_TIME + 1)
eda_features_exct = EDAFeatures(FS_GSR)
ppg_features_exct = PPGFeatures(FS_GSR)
min_eda_len = (FS_GSR * SPLIT_TIME) - 50

for folder in glob.glob(DATASET_PATH + "*"):
    for subject in glob.glob(folder + "\\*-2020-*"):
        print(subject)
        try:
            os.makedirs(subject + EDA_PATH, exist_ok=True)
            os.makedirs(subject + PPG_PATH, exist_ok=True)
            data_EmotionTest = pd.read_csv(glob.glob(subject + game_result)[0])
            eda_data = pd.read_csv(subject + gsr_file + "filtered_eda.csv")
            ppg_data = pd.read_csv(subject + gsr_file + "filtered_ppg.csv")
            eda_data.iloc[:, 0] = eda_data.iloc[:, 0].apply(timeToInt)
            ppg_data.iloc[:, 0] = ppg_data.iloc[:, 0].apply(timeToInt)

            data_EmotionTest["Time_Start"] = data_EmotionTest["Time_Start"].apply(timeToInt)
            data_EmotionTest["Time_End"] = data_EmotionTest["Time_End"].apply(timeToInt)

            gsr_features = pd.DataFrame(
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

                    eda = eda_data[(eda_data.iloc[:, 0].values >= start) & (
                            eda_data.iloc[:, 0].values <= end)]["eda"].values[:min_eda_len]
                    ppg = ppg_data[(ppg_data.iloc[:, 0].values >= start) & (
                            ppg_data.iloc[:, 0].values <= end)]["ppg"].values[:min_eda_len]

                    status = 0
                    # extract eda features
                    # print(eda.shape)
                    if (eda.shape[0] == min_eda_len):
                        # extract cvx of eda and time domain of ppg to check whether the inputs are not disorted
                        cvx_features = eda_features_exct.extractCVXEDA(eda)
                        scr_features = eda_features_exct.extractSCRFeatures(eda)
                        ppg_time = ppg_features_exct.extractTimeDomain(ppg)
                        if (cvx_features.shape[0] != 0) and (ppg_time.shape[0] != 0) and (scr_features.shape[0] != 0):
                            eda_features = np.concatenate(
                                [cvx_features, eda_features_exct.extractMFCCFeatures(eda, min_len=min_len),
                                 scr_features])

                            # extract PPG features
                            ppg_features = np.concatenate(
                                [ppg_time, ppg_features_exct.extractFrequencyDomain(ppg),
                                 ppg_features_exct.extractNonLinear(ppg)])

                            if np.sum(np.isinf(ppg_features)) == 0 and np.sum(np.isnan(ppg_features)) == 0 and np.sum(
                                    np.isinf(eda_features)) == 0 and np.sum(np.isnan(eda_features)) == 0:
                                # print(eda_features.shape)
                                # save features
                                np.save(subject + EDA_PATH + "eda_" + str(idx) + ".npy", eda_features)
                                np.save(subject + PPG_PATH + "ppg_" + str(idx) + ".npy", ppg_features)
                                # np.save(subject + EDA_RAW_PATH +"eda_" + str(idx) + ".npy", eda )
                                # np.save(subject + PPG_RAW_PATH + "ppg_" + str(idx) + ".npy", ppg)
                                status = 1
                            else:
                                status = 0
                        else:
                            status = 0

                    # add object to dataframes
                    subject_name = subject.split("\\")[-1]
                    gsr_features = gsr_features.append(
                        {"Idx": str(idx), "Subject": subject_name, "Start": str(start), "End": str(end),
                         "Valence": valence, "Arousal": arousal, "Emotion": emotion, "Status": status},
                        ignore_index=True)
                    idx += 1

            # save to csv
            gsr_features.to_csv(subject + "\\GSR_features_list_" + str(STRIDE) + ".csv", index=False)
        except ValueError:
            print("Error" + subject)
