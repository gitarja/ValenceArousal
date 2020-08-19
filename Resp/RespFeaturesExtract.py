from Resp.RespFeatures import RespFeatures
import pandas as pd
from datetime import datetime
from Libs.Utils import timeToInt, utcToTimeStamp
from Conf.Settings import FS_RESP
import numpy as np

path = "D:\\usr\\pras\\data\\EmotionTestVR\\Okada\\"
path_results = path + "results\\Resp\\"
experiment_results = pd.read_csv(path + "Okada_M_2020_7_30_17_5_5_gameResults.csv")
resp_data = pd.read_csv(path + "Okada_Resp.csv", header=[0, 1])
resp_data["RESP_Timestamp_Unix_CAL"] = resp_data["RESP_Timestamp_Unix_CAL"].apply(utcToTimeStamp, axis=1)
experiment_results["Time_Start"] = experiment_results["Time_Start"].apply(timeToInt)
experiment_results["Time_End"] = experiment_results["Time_End"].apply(timeToInt)

format = '%H:%M:%S'
split_time = 45

resp_features_exct = RespFeatures(FS_RESP)
min_len = FS_RESP * (split_time + 1)

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
        resp_split = resp_data[(resp_data["RESP_Timestamp_Unix_CAL"].values >= start) & (
                resp_data["RESP_Timestamp_Unix_CAL"].values <= end)]
        resp = resp_split["RESP_ECG_RESP_24BIT_CAL"].values.flatten()

        status = 0
        # extract resp features
        resp_time_features = resp_features_exct.extractTimeDomain(resp)
        if (resp_time_features.shape[0] !=0):
            resp_features = np.concatenate([resp_time_features, resp_features_exct.extractFrequencyDomain(resp), resp_features_exct.extractNonLinear(resp)])
            if (np.sum(np.isinf(resp_features)) == 0 | np.sum(np.isnan(resp_features)) == 0):
                np.save(path_results + "resp_" + str(idx) + ".npy", resp_features)
                status = 1
            else:
                status = 0

        # add object to dataframes
        gsr_features = gsr_features.append(
            {"Idx": str(idx), "Start":str(start), "End": str(end), "Valence": valence, "Arousal": arousal, "Emotion": emotion, "Status": status},
            ignore_index=True)
        idx+=1


gsr_features.to_csv(path+"Resp_features_list.csv")
