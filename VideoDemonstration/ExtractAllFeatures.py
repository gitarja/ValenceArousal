from EEG.EEGFeatures import EEGFeatures
from Resp.RespFeatures import RespFeatures
from ECG.ECGFeatures import ECGFeatures, ECGRespFeatures
import pandas as pd
import glob
from Libs.Utils import timeToInt, arToLabels
from Conf.Settings import FS_EEG, SPLIT_TIME, STRIDE, EXTENTION_TIME, EEG_RAW_PATH, EEG_PATH, DATASET_PATH, ALL_FEATURES_PATH, FS_GSR, FS_ECG, FS_RESP
from EEG.SpaceLapFilter import SpaceLapFilter
import numpy as np
from GSR.GSRFeatures import PPGFeatures, EDAFeatures
from os import path
import os

eeg_file = "\\EEG\\"
ecg_file = "\\ECG\\"
gsr_file = "\\GSR\\"
resp_file = "\\Resp\\"
game_result = "\\*_gameResults.csv"
min_eeg_len = SPLIT_TIME * FS_EEG - 100
downsample_eeg_len = SPLIT_TIME * 200
#EDA Features extractor
eda_features_exct = EDAFeatures(FS_GSR)
ppg_features_exct = PPGFeatures(FS_GSR)
min_eda_len = (FS_GSR * SPLIT_TIME) - 50

#EEG features extractor
eeg_filter = SpaceLapFilter()
eeg_features_exct = EEGFeatures(FS_EEG)

#resp features extractor
resp_features_exct = RespFeatures(FS_RESP)
ecg_resp_features_exct = ECGRespFeatures(FS_RESP)

#ECG features exract
ecg_features_exct = ECGFeatures(FS_ECG)

for folder in glob.glob(DATASET_PATH + "*"):

    for subject in glob.glob(folder + "\\*-2020-*"):
        if not path.exists(subject + ALL_FEATURES_PATH):
            os.mkdir(subject + ALL_FEATURES_PATH)
        try:
            data_EmotionTest = pd.read_csv(glob.glob(subject + game_result)[0])
            #open ECG
            ecg_data = pd.read_csv(glob.glob(subject + ecg_file + "*.csv")[0])
            ecg_data.loc[:, 'timestamp'] = ecg_data.loc[:, 'timestamp'].apply(timeToInt)
            #open GSR
            eda_data = pd.read_csv(subject + gsr_file + "filtered_eda.csv")
            ppg_data = pd.read_csv(subject + gsr_file + "filtered_ppg.csv")
            eda_data.iloc[:, 0] = eda_data.iloc[:, 0].apply(timeToInt)
            ppg_data.iloc[:, 0] = ppg_data.iloc[:, 0].apply(timeToInt)
            #open RESP
            resp_data = pd.read_csv(subject + resp_file + "filtered_resp.csv")
            ecg_resp_data = pd.read_csv(subject + resp_file + "filtered_ecg_resp.csv")
            resp_data.iloc[:, 0] = resp_data.iloc[:, 0].apply(timeToInt)
            ecg_resp_data.iloc[:, 0] = ecg_resp_data.iloc[:, 0].apply(timeToInt)

            data_EmotionTest.loc[:, 'Time_Start'] = data_EmotionTest.loc[:, 'Time_Start'].apply(timeToInt)
            data_EmotionTest.loc[:, 'Time_End'] = data_EmotionTest.loc[:, 'Time_End'].apply(timeToInt)

            video_features_list = pd.DataFrame(
                columns=["Idx", "Start", "End", "Valence", "Arousal", "Emotion", "Status", "Subject"])
            idx = 0

            for i in range(len(data_EmotionTest)):

                time_start = data_EmotionTest.iloc[i]["Time_Start"]
                time_end = data_EmotionTest.iloc[i]["Time_End"]
                valence = data_EmotionTest.iloc[i]["Valence"]
                arousal = data_EmotionTest.iloc[i]["Arousal"]
                emotion = data_EmotionTest.iloc[i]["Emotion"]
                eeg = pd.read_csv(subject + eeg_file + "filtered_eeg" + str(i) + ".csv")
                print(i)
                eeg["time"] = eeg["time"].apply(timeToInt)
                for j in np.arange(time_start, time_end, 1.): #the window move backward



                    end = time_end - ((j+1) * SPLIT_TIME)+ EXTENTION_TIME
                    start = time_end - ((j) * SPLIT_TIME)

                    #split EEG
                    eeg_split = eeg[(eeg["time"].values >= start) & (
                            eeg["time"].values <= end)]
                    eeg_filtered = eeg_split.loc[:, "CH0":"CH18"].values[:min_eeg_len]

                    #split resp
                    resp = resp_data[(resp_data.iloc[:, 0].values >= start) & (
                            resp_data.iloc[:, 0].values <= end)]["resp"].values
                    ecg_resp = ecg_resp_data[(ecg_resp_data.iloc[:, 0].values >= start) & (
                            ecg_resp_data.iloc[:, 0].values <= end)]["ecg"].values

                    #split GSR
                    eda = eda_data[(eda_data.iloc[:, 0].values >= start) & (
                            eda_data.iloc[:, 0].values <= end)]["eda"].values[:min_eda_len]
                    ppg = ppg_data[(ppg_data.iloc[:, 0].values >= start) & (
                            ppg_data.iloc[:, 0].values <= end)]["ppg"].values[:min_eda_len]


                    #split ECG
                    ecg = ecg_data[(ecg_data["timestamp"].values >= start) & (ecg_data["timestamp"].values <= end)]
                    ecg_values = ecg['ecg'].values
                    status = 0

                    if eeg_filtered.shape[0] >= min_eeg_len and (eda.shape[0] == min_eda_len) :
                        # Extract EEG Features
                        time_domain_features = eeg_features_exct.extractTimeDomainAll(eeg_filtered)
                        freq_domain_features = eeg_features_exct.extractFrequencyDomainAll(eeg_filtered)
                        plf_features = eeg_features_exct.extractPLFFeatures(eeg_filtered)
                        power_features = eeg_features_exct.extractPowerFeatures(eeg_filtered)
                        eeg_features = np.concatenate([time_domain_features, freq_domain_features, plf_features, power_features])

                        # extract cvx of eda and time domain of ppg to check whether the inputs are not disorted
                        cvx_features = eda_features_exct.extractCVXEDA(eda)
                        scr_features = eda_features_exct.extractSCRFeatures(eda)
                        ppg_time = ppg_features_exct.extractTimeDomain(ppg)
                        eda_features = np.concatenate(
                            [cvx_features, eda_features_exct.extractMFCCFeatures(eda, min_len=min_eda_len), scr_features])
                        ppg_features = np.concatenate(
                            [ppg_time, ppg_features_exct.extractFrequencyDomain(ppg),
                             ppg_features_exct.extractNonLinear(ppg)])

                        # extract ecg resp features
                        ecg_resp_time_domain = ecg_resp_features_exct.extractTimeDomain(ecg_resp)
                        ecg_resp_freq_domain = ecg_resp_features_exct.extractFrequencyDomain(ecg_resp)
                        ecg_resp_nonlinear_domain = ecg_resp_features_exct.extractNonLinearDomain(ecg_resp)
                        ecg_resp_features = np.concatenate([ecg_resp_time_domain, ecg_resp_freq_domain, ecg_resp_nonlinear_domain])
                        # extract resp features
                        resp_time_features = resp_features_exct.extractTimeDomain(resp)
                        resp_freq_features = resp_features_exct.extractFrequencyDomain(resp)
                        resp_nonlinear_features = resp_features_exct.extractNonLinear(resp)
                        resp_features = np.concatenate(
                            [resp_time_features, resp_features_exct.extractFrequencyDomain(resp),
                             resp_features_exct.extractNonLinear(resp)])

                        # extract ecg features
                        ecg_time_domain = ecg_features_exct.extractTimeDomain(ecg_values)
                        ecg_freq_domain = ecg_features_exct.extractFrequencyDomain(ecg_values)
                        ecg_nonlinear_domain = ecg_features_exct.extractNonLinearDomain(ecg_values)
                        ecg_features = np.concatenate([ecg_time_domain, ecg_freq_domain, ecg_nonlinear_domain])

                        features = [eda_features, ppg_features, resp_features, ecg_resp_features, ecg_features, eeg_features]

                        np.save(subject + ALL_FEATURES_PATH + "features" + str(idx) + ".npy", features)


                        # add object to dataframes
                    subject_name = subject.split("\\")[-1]
                    video_features_list = video_features_list.append(
                            {"Idx": str(idx), "Subject": subject_name, "Start": str(start), "End": str(end),
                             "Valence": valence, "Arousal": arousal,
                             "Emotion": emotion},
                            ignore_index=True)
                    idx += 1

            video_features_list.to_csv(subject + "\\video_features_list_"+str(STRIDE)+".csv", index=False)

        except:
            print("Error: " + subject)
