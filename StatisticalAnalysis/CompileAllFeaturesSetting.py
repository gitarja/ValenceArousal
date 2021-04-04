import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from Libs.Utils import arToLabels, valToLabels, convertLabels, regressLabelsConv, convertLabelsReg
import glob
import numpy as np
from Conf.Settings import DATASET_PATH, STRIDE, ECG_PATH

for folder in glob.glob(DATASET_PATH + "2020-*"):
    for subject in glob.glob(folder + "\\*-2020-*"):
        try:
            eeg_features_list = pd.read_csv(subject + "\\EEG_features_list_" + str(STRIDE) + ".csv").set_index('Idx')
            ecg_features_list = pd.read_csv(subject + "\\ECG_features_list_" + str(STRIDE) + ".csv").set_index('Idx')
            GSR_features_list = pd.read_csv(subject + "\\GSR_features_list_" + str(STRIDE) + ".csv").set_index('Idx')
            Resp_features_list = pd.read_csv(subject + "\\Resp_features_list_" + str(STRIDE) + ".csv").set_index('Idx')

            features_list = ecg_features_list[
                (eeg_features_list["Status"] == 1) & (GSR_features_list["Status"] == 1) & (
                            Resp_features_list["Status"] == 1) & (ecg_features_list["Status"] == 1)]
            # print(str(np.sum(eeg_features_list["Status"].values)) + "," + str(len(features_list)))
            # features_list = ecg_features_list[ (GSR_features_list["Status"] == 1) & (
            #                 Resp_features_list["Status"] == 1) & (ecg_features_list["Status"] == 1)]
            features_list.to_csv(subject + "\\features_list_" + str(STRIDE) + ".csv")
        except:
            print("Error: " + subject)

# SPlit to train test and val

all_features = []
ecg_mean = np.load("G:\\usr\\nishihara\\GitHub\\EmotionRecognition\\Values\\ecg_mean.npy")
for folder in glob.glob(DATASET_PATH + "2020-*"):
    for subject in glob.glob(folder + "\\*-2020-*"):
        features_list = pd.read_csv(subject + "\\features_list_" + str(STRIDE) + ".csv")
        features_list_ori = pd.read_csv(subject + "\\features_list_" + str(STRIDE) + ".csv")
        video_seq = 0
        ecg_features_all = []
        weights = []

        for idx in features_list["Idx"].values:
            ecg_features_all.append(np.expand_dims(np.load(subject + ECG_PATH + "ecg_" + str(idx) + ".npy"), axis=0))
        ecg_features_all = np.concatenate(ecg_features_all, axis=0)
        hr_mean = np.mean(ecg_features_all[:, 6])

        ecg_features = []
        for i, idx in enumerate(features_list["Idx"].values):
            ecg_features.append(np.load(subject + ECG_PATH + "ecg_" + str(idx) + ".npy"))
            if (i + 1) < len(features_list):
                t_delta = features_list.iloc[i + 1]["Start"] - features_list.iloc[i]["Start"]

            if (t_delta >= 0) or ((i + 1) >= len(features_list)):
                if len(ecg_features) < 10:
                    weights.append(np.ones((len(ecg_features),)))
                else:
                    for f in ecg_features:
                        if f[6] < hr_mean:
                            weights.append(np.array([0.5]))
                        else:
                            weights.append(np.array([1.]))

                ecg_features = []

        weights = np.concatenate(weights, axis=0)
        features_list["Weight"] = weights
        features_list["Valence_convert"] = features_list["Valence"].apply(arToLabels)
        features_list["Arousal_convert"] = features_list["Arousal"].apply(valToLabels)
        all_features.append(features_list)

df = pd.concat(all_features, ignore_index=True)

# resample
# df_postivie_ar = df[(df["Arousal_convert"].values == 1) & (df["Valence_convert"].values == 1) ].sample(frac=.7)
# df_postivie_val = df[(df["Arousal_convert"].values == 1) & (df["Valence_convert"].values == 0) ].sample(frac=1.)
#
# df_negative_ar = df[(df["Arousal_convert"].values == 0) & (df["Valence_convert"].values == 1) ].sample(frac=1.)
# df_negative_val = df[(df["Arousal_convert"].values == 0) & (df["Valence_convert"].values == 0) ].sample(frac=1.)
# #
# df = pd.concat([df_postivie_ar, df_postivie_val, df_negative_ar, df_negative_val])

# compute label porpotion

ar_labels = df["Arousal_convert"].values
val_labels = df["Valence_convert"].values

# #arousal
# ar_positive = 1 / np.sum(ar_labels==1) * (len(ar_labels) / 2.)
# ar_negative = 1 / np.sum(ar_labels==0) * (len(ar_labels) / 2.)
# #valence
# val_positive = 1 / np.sum(val_labels==1) * (len(val_labels) / 2.)
# val_negative = 1 / np.sum(val_labels==0) * (len(val_labels) / 2.)
#
#
# print(ar_positive)
# print(ar_negative)
# print(val_positive)
# print(val_negative)

y = convertLabelsReg(df["Arousal_convert"].values, df["Valence_convert"].values)

# #Split to train and test
skf = StratifiedKFold(n_splits=5, shuffle=True)
fold = 1
for train_index, test_index in skf.split(df.index, y):
    X_val, X_test, _, _ = train_test_split(test_index, y[test_index], test_size=0.5, random_state=0)

    # #training data
    training_data = df.iloc[train_index]
    val_data = df.iloc[X_val]
    test_data = df.iloc[X_test]
    #
    training_data.to_csv(DATASET_PATH + "stride=" + str(STRIDE) + "\\training_data_" + str(fold) + ".csv", index=False)
    val_data.to_csv(DATASET_PATH + "stride=" + str(STRIDE) + "\\validation_data_" + str(fold) + ".csv", index=False)
    test_data.to_csv(DATASET_PATH + "stride=" + str(STRIDE) + "\\test_data_" + str(fold) + ".csv", index=False)
    fold += 1
