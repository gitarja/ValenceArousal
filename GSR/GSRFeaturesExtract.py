from GSR.GSRFeatures import PPGFeatures, EDAFeatures
import pandas as pd
from datetime import datetime
from Libs.Utils import timeToInt, utcToTimeStamp
from Conf.Settings import FS_GSR
import numpy as np

path = "D:\\usr\\pras\\data\\EmotionTestVR\\Komiya\\"
path_results = path + "results\\GSR\\"
experiment_results = pd.read_csv(path + "Komiya_M_2020_7_9_15_22_44_gameResults.csv")
gsr_data = pd.read_csv(path + "Komiya_GSR.csv", header=[0, 1])
gsr_data["GSR_Timestamp_Unix_CAL"] = gsr_data["GSR_Timestamp_Unix_CAL"].apply(utcToTimeStamp, axis=1)
experiment_results["Time_Start"] = experiment_results["Time_Start"].apply(timeToInt)
experiment_results["Time_End"] = experiment_results["Time_End"].apply(timeToInt)

format = '%H:%M:%S'
split_time = 45

eda_features_exct = EDAFeatures(FS_GSR)
ppg_features_exct = PPGFeatures(FS_GSR)
min_len = FS_GSR * (split_time + 1)

gsr_features = pd.DataFrame(columns=["Idx", "Start", "End", "Valence", "Arousal", "Emotion", "Status"])
idx = 0
for i in range(len(experiment_results)):
    tdelta = experiment_results.iloc[i]["Time_End"] - experiment_results.iloc[i]["Time_Start"]
    time_end = experiment_results.iloc[i]["Time_End"]
    valence = experiment_results.iloc[i]["Valence"]
    arousal = experiment_results.iloc[i]["Arousal"]
    emotion = experiment_results.iloc[i]["Emotion"]

    for j in np.arange(0, (tdelta // split_time), 0.1):
        end = time_end - (j * split_time)
        start = time_end - ((j + 1) * split_time)
        gsr_split = gsr_data[(gsr_data["GSR_Timestamp_Unix_CAL"].values >= start) & (
                gsr_data["GSR_Timestamp_Unix_CAL"].values <= end)]
        eda = gsr_split["GSR_GSR_Skin_Conductance_CAL"].values.flatten()
        ppg = gsr_split["GSR_PPG_A13_CAL"].values.flatten()

        status = 0
        # extract eda features
        #scr_features = eda_features_exct.extractSCRFeatures(eda)

        #extract cvx of eda and time domain of ppg to check whether the inputs are not disorted
        cvx_features = eda_features_exct.extractCVXEDA(eda)
        ppg_time = ppg_features_exct.extractTimeDomain(ppg)
        if (cvx_features.shape[0] != 0) & (ppg_time.shape[0] != 0):
            eda_features = np.concatenate([cvx_features, eda_features_exct.extractMFCCFeatures(eda, min_len=min_len)])

            # extract PPG features
            ppg_features = np.concatenate(
                [ppg_time, ppg_features_exct.extractFrequencyDomain(ppg),
                 ppg_features_exct.extractNonLinear(ppg)])

            if (np.sum(np.isinf(ppg_features))== 0 | np.sum(np.isnan(ppg_features)) == 0):
                # print(eda_features.shape)
                # save features
                np.save(path_results + "eda_" + str(idx) + ".npy", eda_features)
                np.save(path_results + "ppg_" + str(idx) + ".npy", ppg_features)
                status = 1
            else:
                status = 0
        else:
            status = 0

        # add object to dataframes
        gsr_features = gsr_features.append(
            {"Idx": str(idx), "Start":str(start), "End": str(end), "Valence": valence, "Arousal": arousal, "Emotion": emotion, "Status": status},
            ignore_index=True)
        idx+=1


gsr_features.to_csv(path+"GSR_features_list.csv")
