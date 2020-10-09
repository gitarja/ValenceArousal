from EEG.EEGFeatures import EEGFeatures
import pandas as pd

from Libs.Utils import timeToInt
from Conf.Settings import FS_EEG
from EEG.SpaceLapFilter import SpaceLapFilter
import numpy as np

subject = "Komiya"
path = "D:\\usr\\pras\\data\\EmotionTestVR\\"+subject+"\\"
path_results = path + "results\\EEG\\"
experiment_results = pd.read_csv(path + "Komiya_M_2020_7_9_15_22_44_gameResults.csv")

experiment_results["Time_Start"] = experiment_results["Time_Start"].apply(timeToInt)
experiment_results["Time_End"] = experiment_results["Time_End"].apply(timeToInt)

format = '%H:%M:%S'
split_time = 45


eeg_features_list = pd.DataFrame(columns=["Idx", "Start", "End", "Valence", "Arousal", "Emotion", "Status",  "Subject"])
idx = 0
eeg_filter = SpaceLapFilter()
eeg_features_exct = EEGFeatures(FS_EEG)
for i in range(len(experiment_results)):
    tdelta = experiment_results.iloc[i]["Time_End"] - experiment_results.iloc[i]["Time_Start"]
    time_end = experiment_results.iloc[i]["Time_End"]
    valence = experiment_results.iloc[i]["Valence"]
    arousal = experiment_results.iloc[i]["Arousal"]
    emotion = experiment_results.iloc[i]["Emotion"]
    eeg_data = pd.read_csv(path+"eeg\\eeg"+str(i)+".csv")
    print(i)
    eeg_data["Timestamp_Unix_CAL"] = eeg_data["Timestamp_Unix_CAL"].apply(timeToInt)

    eeg_data.loc[:, "CH1":"CH19"] = eeg_filter.FilterEEG(eeg_data.loc[:, "CH1":"CH19"].values, mode=4)

    for j in np.arange(0, (tdelta // split_time), 0.4):
        end = time_end - (j * split_time)
        start = time_end - ((j + 1) * split_time)
        eeg_split = eeg_data[(eeg_data["Timestamp_Unix_CAL"].values >= start) & (
                eeg_data["Timestamp_Unix_CAL"].values <= end)]
        eeg_filtered = eeg_split.loc[:, "CH1":"CH19"].values
        # eeg_filtered = eeg_filter.FilterEEG(eeg, mode=4)
        time_domain_features = eeg_features_exct.extractTimeDomainAll(eeg_filtered)
        freq_domain_features = eeg_features_exct.extractTimeDomainAll(eeg_filtered)


        if (time_domain_features.shape[0] != 0) & (freq_domain_features.shape[0] != 0):
            eeg_features = np.concatenate([time_domain_features, freq_domain_features])
            np.save(path_results + "eeg_" + str(idx) + ".npy", eeg_features)
            status = 1

        # add object to dataframes
        eeg_features_list = eeg_features_list.append(
                {"Idx": str(idx), "Subject": subject, "Start": str(start), "End": str(end), "Valence": valence, "Arousal": arousal,
                 "Emotion": emotion, "Status": status},
                ignore_index=True)
        idx += 1

eeg_features_list.to_csv(path + "EEG_features_list.csv")



