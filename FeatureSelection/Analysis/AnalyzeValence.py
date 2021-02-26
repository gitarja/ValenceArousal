import numpy as np
import pandas as pd
from Libs.Utils import valArLevelToLabels
from FeatureSelection.mifs.mifs import MutualInformationFeatureSelector
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import glob
import os


def fitJMIM(x, y, selector, i):
    selector.fit(x, y)
    return selector, i


STRIDE = 0.1
eda_features = []
ppg_features = []
resp_features = []
eeg_features = []
ecg_features = []
ecg_resp_features = []
y_val = []

data_path = "G:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\2020-*"
# data_path = "G:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\2020-11-03"

# load features data
print("Loading data...")
for count, folder in enumerate(glob.glob(data_path)):
    print("{}/{}".format(count + 1, len(glob.glob(data_path))) + "\r", end="")
    for subject in glob.glob(folder + "\\*-2020-*"):
        eeg_path = subject + "\\results_stride=" + str(STRIDE) + "\\EEG\\"
        eda_path = subject + "\\results_stride=" + str(STRIDE) + "\\eda\\"
        ppg_path = subject + "\\results_stride=" + str(STRIDE) + "\\ppg\\"
        resp_path = subject + "\\results_stride=" + str(STRIDE) + "\\Resp\\"
        ecg_path = subject + "\\results_stride=" + str(STRIDE) + "\\ECG\\"
        ecg_resp_path = subject + "\\results_stride=" + str(STRIDE) + "\\ECG_resp\\"

        features_list = pd.read_csv(subject + "\\features_list_" + str(STRIDE) + ".csv")
        features_list["Valence"] = features_list["Valence"].apply(valArLevelToLabels)
        for i in range(len(features_list)):
            filename = features_list.iloc[i]["Idx"]
            eda_features.append(np.load(eda_path + "eda_" + str(filename) + ".npy"))
            ppg_features.append(np.load(ppg_path + "ppg_" + str(filename) + ".npy"))
            resp_features.append(np.load(resp_path + "resp_" + str(filename) + ".npy"))
            eeg_features.append(np.load(eeg_path + "eeg_" + str(filename) + ".npy"))
            ecg_features.append(np.load(ecg_path + "ecg_" + str(filename) + ".npy"))
            ecg_resp_features.append(np.load(ecg_resp_path + "ecg_resp_" + str(filename) + ".npy"))
            y_val.append(features_list.iloc[i]["Valence"])

            # concat_features = np.concatenate(
            #     [eda_features, ppg_features, resp_features, ecg_features, ecg_resp_features, eeg_features])
            # if np.sum(np.isinf(concat_features)) == 0 & np.sum(np.isnan(concat_features)) == 0:
            #     # print(eda_features.shape)
            #     # print(concat_features.shape)
            #     features.append(concat_features)
            #     y_ar.append(features_list.iloc[i]["Arousal"])
            #     y_val.append(features_list.iloc[i]["Valence"])
            # else:
            #     print(subject + "_" + str(i))

# subject = data_path + "\\B3-2020-11-03"
# eeg_path = subject + "\\results\\eeg\\"
# eda_path = subject + "\\results\\eda\\"
# ppg_path = subject + "\\results\\ppg\\"
# resp_path = subject + "\\results\\resp\\"
# ecg_path = subject + "\\results\\ecg\\"
# ecg_resp_path = subject + "\\results\\ecg_resp\\"
#
# features_list = pd.read_csv(subject + "\\features_list.csv")
# features_list["Valence"] = features_list["Valence"].apply(valArLevelToLabels)
# features_list["Arousal"] = features_list["Arousal"].apply(valArLevelToLabels)
# for i in range(len(features_list)):
#     filename = features_list.iloc[i]["Idx"]
#     eda_features.append(np.load(eda_path + "eda_" + str(filename) + ".npy"))
#     ppg_features.append(np.load(ppg_path + "ppg_" + str(filename) + ".npy"))
#     resp_features.append(np.load(resp_path + "resp_" + str(filename) + ".npy"))
#     eeg_features.append(np.load(eeg_path + "eeg_" + str(filename) + ".npy"))
#     ecg_features.append(np.load(ecg_path + "ecg_" + str(filename) + ".npy"))
#     ecg_resp_features.append(np.load(ecg_resp_path + "ecg_resp_" + str(filename) + ".npy"))
#     y_ar.append(features_list.iloc[i]["Arousal"])
#
#     concat_features = np.concatenate(
#         [eda_features, ppg_features, resp_features, ecg_features, ecg_resp_features, eeg_features])
#     if np.sum(np.isinf(concat_features)) == 0 & np.sum(np.isnan(concat_features)) == 0:
#         # print(eda_features.shape)
#         # print(concat_features.shape)
#         features.append(concat_features)
#         y_ar.append(features_list.iloc[i]["Arousal"])
#         y_val.append(features_list.iloc[i]["Valence"])
#     else:
#         print(subject + "_" + str(i))

features = []
features.append(np.array(eda_features))
features.append(np.array(ppg_features))
features.append(np.array(resp_features))
features.append(np.array(ecg_features))
features.append(np.array(ecg_resp_features))
features.append(np.array(eeg_features))

print("Finish")
print("EDA Features:", eda_features[0].shape[0])
print("PPG Features:", ppg_features[0].shape[0])
print("Resp Features:", resp_features[0].shape[0])
print("ECG Features:", ecg_features[0].shape[0])
print("ECG Resp Features:", ecg_resp_features[0].shape[0])
print("EEG Features:", eeg_features[0].shape[0])

# normalize features
scaler = StandardScaler()
len_features = 0
for i, feature in enumerate(features):
    features[i] = scaler.fit_transform(feature)
    len_features += feature.shape[1]  # count number of data
len_data = features[0].shape[0]

y_val = np.array(y_val)

print("All Features;", len_features)
print("Number of Data:", len_data)

# Define Feature Selector
feature_selector = []
for f in features:
    selector = MutualInformationFeatureSelector(method='JMIM',
                                                k=5,
                                                n_features=f.shape[1],
                                                categorical=True,
                                                n_jobs=-1,
                                                verbose=2)
    feature_selector.append(selector)

# Analyze features
print("Analyzing Features...")
feature_selector = Parallel(n_jobs=-1)(
    [delayed(fitJMIM)(x, y_val, s, i) for i, (x, s) in enumerate(zip(features, feature_selector))])
feature_selector.sort(key=lambda x: x[1])
feature_selector = [s[0] for s in feature_selector]

# Save result
path_result = "G:\\usr\\nishihara\\data\\Yamaha-Experiment\\jmim_results\\"
os.makedirs(path_result, exist_ok=True)
# print(feature_selector.ranking_, feature_selector.mi_)
results_jmi = []
results_ranking = []
for i, s in enumerate(feature_selector):
    result = np.zeros(len(s.mi_))
    result[s.ranking_] = s.mi_
    results_jmi.append(result)
    results_ranking.append(s.ranking_)

result_df = pd.DataFrame(results_jmi, index=["JMI_EDA", "JMI_PPG", "JMI_Resp", "JMI_ECG", "JMI_ECG_Resp", "JMI_EEG"]).T
result_df_ranking = pd.DataFrame(results_ranking,
                                 index=["Ranking_EDA", "Ranking_PPG", "Ranking_Resp", "Ranking_ECG", "Ranking_ECG_Resp",
                                        "Ranking_EEG"]).T
result_df.to_csv(path_result + "jmim_feature_analysis_ar_" + str(STRIDE) + ".csv", index=False)
result_df_ranking.to_csv(path_result + "jmim_feature_ranking_ar_" + str(STRIDE) + ".csv", index=False)
