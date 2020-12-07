import pandas as pd
from sklearn.model_selection import train_test_split
from Libs.Utils import valArLevelToLabels, convertLabels
import glob
import numpy as np

data_path = "D:\\usr\\pras\\data\\YAMAHA\\Yamaha-Experiment (2020-10-26 - 2020-11-06)\\data\\*"

for folder in glob.glob(data_path):
    for subject in glob.glob(folder + "\\*-2020-*"):
        try:
            eeg_features_list = pd.read_csv(subject+"\\EEG_features_list_1.0.csv").set_index('Idx')
            ecg_features_list = pd.read_csv(subject+"\\ECG_features_list_1.0.csv").set_index('Idx')
            GSR_features_list = pd.read_csv(subject+"\\GSR_features_list_1.0.csv").set_index('Idx')
            Resp_features_list = pd.read_csv(subject+"\\Resp_features_list_1.0.csv").set_index('Idx')

            features_list = ecg_features_list[(eeg_features_list["Status"]==1) & (GSR_features_list["Status"]==1) & (Resp_features_list["Status"]==1) & (ecg_features_list["Status"]==1)]
            # features_list = ecg_features_list[ (GSR_features_list["Status"] == 1) & (
            #                 Resp_features_list["Status"] == 1) & (ecg_features_list["Status"] == 1)]
            features_list.to_csv(subject+"\\features_list_1.0.csv")
        except:
            print("Error: "+ subject)


# SPlit to train test and val

# data_path = "D:\\usr\\pras\\data\\YAMAHA\\Yamaha-Experiment (2020-10-26 - 2020-11-06)\\data\\"
# all_features = []
# all_ori_features = []
# for folder in glob.glob(data_path + "*"):
#     for subject in glob.glob(folder + "\\*-2020-*"):
#         try:
#             features_list = pd.read_csv(subject + "\\features_list.csv")
#             features_list_ori = pd.read_csv(subject + "\\features_list.csv")
#             features_list["Valence"] = features_list["Valence"].apply(valArLevelToLabels)
#             features_list["Arousal"] = features_list["Arousal"].apply(valArLevelToLabels)
#             all_features.append(features_list)
#             all_ori_features.append(features_list_ori)
#         except:
#             print("Error" + subject)

# df = pd.concat(all_features)
# df_ori = pd.concat(all_ori_features)
# y = convertLabels(df["Arousal"].values, df["Valence"].values)
#
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

# #Split to train and test
# X_train, X_test, y_train, y_test = train_test_split(df.index, y, test_size=0.4, random_state=42)
#
#
# X_val, X_test, _, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
#
# #training data
# training_data = df_ori.iloc[X_train]
# val_data = df_ori.iloc[X_val]
# test_data = df_ori.iloc[X_test]
#
# training_data.to_csv(data_path + "training_data.csv", index=False)
# val_data.to_csv(data_path + "validation_data.csv", index=False)
# test_data.to_csv(data_path + "test_data.csv", index=False)
