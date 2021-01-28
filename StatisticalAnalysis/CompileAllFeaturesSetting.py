import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from Libs.Utils import arToLabels, valToLabels, convertLabels
import glob
import numpy as np
from Conf.Settings import DATASET_PATH, STRIDE

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
            # features_list = ecg_features_list[ (GSR_features_list["Status"] == 1) & (
            #                 Resp_features_list["Status"] == 1) & (ecg_features_list["Status"] == 1)]
            features_list.to_csv(subject + "\\features_list_" + str(STRIDE) + ".csv")
        except:
            print("Error: " + subject)

# SPlit to train test and val

all_features = []
all_ori_features = []
for folder in glob.glob(DATASET_PATH + "2020-*"):
    for subject in glob.glob(folder + "\\*-2020-*"):
        try:
            features_list = pd.read_csv(subject + "\\features_list_"+str(STRIDE)+".csv")
            features_list_ori = pd.read_csv(subject + "\\features_list_"+str(STRIDE)+".csv")
            features_list["Valence"] = features_list["Valence"].apply(arToLabels)
            features_list["Arousal"] = features_list["Arousal"].apply(valToLabels)
            all_features.append(features_list)
            all_ori_features.append(features_list_ori)
        except:
            print("Error" + subject)

df = pd.concat(all_features, ignore_index=True)
df_ori = pd.concat(all_ori_features, ignore_index=True)
y = convertLabels(df["Arousal"].values, df["Valence"].values)

# #Split to train and test
skf = StratifiedKFold(n_splits=5, shuffle=False)
fold = 1
for train_index, test_index in skf.split(df.index, y):
    X_val, X_test, _, _ = train_test_split(test_index, y[test_index], test_size=0.5, random_state=42)

    # #training data
    training_data = df_ori.iloc[df.index[train_index]]
    val_data = df_ori.iloc[X_val]
    test_data = df_ori.iloc[X_test]
    #
    training_data.to_csv(DATASET_PATH + "training_data_" + str(fold) + ".csv", index=False)
    val_data.to_csv(DATASET_PATH + "validation_data_" + str(fold) + ".csv", index=False)
    test_data.to_csv(DATASET_PATH + "test_data_" + str(fold) + ".csv", index=False)
    fold += 1

# # Split to cross validation
#
# subjects = np.unique(df["Subject"].values)
# np.random.shuffle(subjects)
#
# for j in range(len(subjects) // 6):
#     val_subjects = subjects[j * 6: (j + 1) * 6]
#
#     training_data = df_ori[
#         (df["Subject"] != val_subjects[0]) & (df["Subject"] != val_subjects[1]) & (df["Subject"] != val_subjects[2]) & (
#                     df["Subject"] != val_subjects[3]) & (df["Subject"] != val_subjects[4]) & (
#                     df["Subject"] != val_subjects[5])]
#     val_data = df_ori[(df["Subject"] == val_subjects[0]) | (df["Subject"] == val_subjects[1]) | (
#             df["Subject"] == val_subjects[2])]
#     test_data = df_ori[
#         (df["Subject"] == val_subjects[3]) | (df["Subject"] == val_subjects[4]) | (df["Subject"] == val_subjects[5])]
#
#     training_data.to_csv(data_path + "training_data_" + str(j) + ".csv", index=False)
#     val_data.to_csv(data_path + "validation_data_" + str(j) + ".csv", index=False)
