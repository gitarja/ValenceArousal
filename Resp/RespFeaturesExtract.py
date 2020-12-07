from Resp.RespFeatures import RespFeatures
from ECG.ECGFeatures import ECGFeatures
import pandas as pd
import glob
from Libs.Utils import timeToInt
from Conf.Settings import FS_RESP, SPLIT_TIME, STRIDE, EXTENTION_TIME, RESP_RAW_PATH, ECG_RAW_RESP_PATH, RESP_PATH, \
    ECG_RESP_PATH
import numpy as np
import os

data_path = "G:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\*"
resp_file = "\\Resp\\"
game_result = "\\*_gameResults.csv"

resp_features_exct = RespFeatures(FS_RESP)
ecg_features_exct = ECGFeatures(FS_RESP)
min_len = FS_RESP * (SPLIT_TIME + 1)

for folder in glob.glob(data_path):
    for subject in glob.glob(folder + "\\*-2020-*"):
        print(subject)
        try:
            os.makedirs(subject + RESP_PATH, exist_ok=True)
            os.makedirs(subject + ECG_RESP_PATH, exist_ok=True)
            data_EmotionTest = pd.read_csv(glob.glob(subject + game_result)[0])
            resp_data = pd.read_csv(subject + resp_file + "filtered_resp.csv")
            ecg_resp_data = pd.read_csv(subject + resp_file + "filtered_ecg_resp.csv")
            resp_data.iloc[:, 0] = resp_data.iloc[:, 0].apply(timeToInt)
            ecg_resp_data.iloc[:, 0] = ecg_resp_data.iloc[:, 0].apply(timeToInt)
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

                    resp = resp_data[(resp_data.iloc[:, 0].values >= start) & (
                            resp_data.iloc[:, 0].values <= end)]["resp"].values
                    ecg_resp = ecg_resp_data[(ecg_resp_data.iloc[:, 0].values >= start) & (
                            ecg_resp_data.iloc[:, 0].values <= end)]["ecg"].values

                    status = 0
                    # extract ecg features
                    time_domain = ecg_features_exct.extractTimeDomain(ecg_resp)
                    freq_domain = ecg_features_exct.extractFrequencyDomain(ecg_resp)
                    nonlinear_domain = ecg_features_exct.extractNonLinearDomain(ecg_resp)
                    # extract resp features
                    resp_time_features = resp_features_exct.extractTimeDomain(resp)
                    resp_freq_features = resp_features_exct.extractFrequencyDomain(resp)
                    resp_nonlinear_features = resp_features_exct.extractNonLinear(resp)
                    if resp_time_features.shape[0] != 0 and resp_freq_features.shape[0] != 0 and \
                            resp_nonlinear_features.shape[0] != 0 and time_domain.shape[0] != 0 and freq_domain.shape[
                        0] != 0 and nonlinear_domain.shape[0] != 0:
                        resp_features = np.concatenate(
                            [resp_time_features, resp_features_exct.extractFrequencyDomain(resp),
                             resp_features_exct.extractNonLinear(resp)])
                        ecg_features = np.concatenate([time_domain, freq_domain, nonlinear_domain])
                        # print(np.sum(np.isinf(ecg_features)))
                        if (np.sum(np.isinf(resp_features)) == 0 and np.sum(np.isnan(resp_features)) == 0 and np.sum(
                                np.isinf(ecg_features)) == 0 and np.sum(np.isnan(ecg_features)) == 0):
                            np.save(subject + RESP_PATH + "resp_" + str(idx) + ".npy", resp_features)
                            np.save(subject + ECG_RESP_PATH + "ecg_resp_" + str(idx) + ".npy", ecg_features)
                            # np.save(subject + RESP_RAW_PATH + "resp_" + str(idx) + ".npy", resp)
                            # np.save(subject + ECG_RAW_RESP_PATH + "ecg_resp_" + str(idx) + ".npy", ecg_resp)
                            status = 1
                        else:
                            status = 0

                    # add object to dataframes
                    subject_name = subject.split("\\")[-1]
                    gsr_features = gsr_features.append(
                        {"Idx": str(idx), "Subject": subject_name, "Start": str(start), "End": str(end),
                         "Valence": valence, "Arousal": arousal,
                         "Emotion": emotion, "Status": status},
                        ignore_index=True)
                    idx += 1

            # save to csv
            gsr_features.to_csv(subject + "\\Resp_features_list_" + str(STRIDE) + ".csv", index=False)
        except:
            print("Error: " + subject)
