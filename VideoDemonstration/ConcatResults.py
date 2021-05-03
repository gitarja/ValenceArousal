import pandas as pd
from Conf.Settings import DATASET_PATH
from Libs.Utils import regressLabelsConv

date = "2020-10-27"
subject = "A6"
path_features_list = DATASET_PATH + date + "\\" + subject + "-" + date + "\\video_features_list_0.2.csv"
path_results_all = DATASET_PATH + date + "\\" + subject + "-" + date + "\\video_results_all_features.csv"
path_results_ecg = DATASET_PATH + date + "\\" + subject + "-" + date + "\\video_results_ecg.csv"

df_features_list = pd.read_csv(path_features_list)
df_results_all = pd.read_csv(path_results_all)
df_results_ecg = pd.read_csv(path_results_ecg)

df_features_list["Valence"] = df_features_list["Valence"].apply(regressLabelsConv)
df_features_list["Arousal"] = df_features_list["Arousal"].apply(regressLabelsConv)
df_features_list["All_features_ar"] = df_results_all["arousal"].values
df_features_list["All_features_val"] = df_results_all["valence"].values
df_features_list["ecg_ar"] = df_results_ecg["arousal"].values
df_features_list["ecg_val"] = df_results_ecg["valence"].values

df_features_list.to_csv(DATASET_PATH + date + "\\" + subject + "-" + date + "\\video_results.csv", index=False)
