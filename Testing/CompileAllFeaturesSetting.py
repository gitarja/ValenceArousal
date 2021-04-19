import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from Libs.Utils import arToLabels, valToLabels, convertLabels, regressLabelsConv, convertLabelsReg
import glob
import numpy as np
from Conf.Settings import DATASET_PATH, STRIDE, ECG_PATH


for folder in glob.glob(DATASET_PATH + "*"):
    for subject in glob.glob(folder + "\\*-2020-*"):
        try:
            eeg_features_list = pd.read_csv(subject+"\\EEG_features_list_"+str(STRIDE)+".csv").set_index('Idx')
            ecg_features_list = pd.read_csv(subject+"\\ECG_features_list_"+str(STRIDE)+".csv").set_index('Idx')
            GSR_features_list = pd.read_csv(subject+"\\GSR_features_list_"+str(STRIDE)+".csv").set_index('Idx')
            Resp_features_list = pd.read_csv(subject+"\\Resp_features_list_"+str(STRIDE)+".csv").set_index('Idx')

            features_list = ecg_features_list[(eeg_features_list["Status"]==1) & (GSR_features_list["Status"]==1) & (Resp_features_list["Status"]==1) & (ecg_features_list["Status"]==1)]
            # print(str(np.sum(eeg_features_list["Status"].values)) + "," + str(len(features_list)))
            # features_list = ecg_features_list[ (GSR_features_list["Status"] == 1) & (
            #                 Resp_features_list["Status"] == 1) & (ecg_features_list["Status"] == 1)]
            features_list.to_csv(subject+"\\features_list_"+str(STRIDE)+".csv")
        except:
            print("Error: "+ subject)


# SPlit to train test and val

all_features = []
for folder in glob.glob(DATASET_PATH + "*"):
    for subject in glob.glob(folder + "\\*-2020-*"):
        try:
            features_list = pd.read_csv(subject + "\\features_list_"+str(STRIDE)+".csv")
            features_list_ori = pd.read_csv(subject + "\\features_list_"+str(STRIDE)+".csv")

            hr_all = []


            for idx in features_list["Idx"].values:
                hr_all.append(
                    np.expand_dims(np.load(subject + ECG_PATH + "ecg_" + str(idx) + ".npy"), axis=0)[0,3])

            features_list["Valence_convert"] = features_list["Valence"].apply(regressLabelsConv)
            features_list["Arousal_convert"] = features_list["Arousal"].apply(regressLabelsConv)
            a_h_v_n = (features_list["Arousal_convert"].values > 1) & (features_list["Valence_convert"].values < 0)
            a_l_v_n = (features_list["Arousal_convert"].values < 0) & (features_list["Valence_convert"].values < 0)
            a_h_v_p = (features_list["Arousal_convert"].values > 0) & (features_list["Valence_convert"].values > 0)
            a_l_v_p = (features_list["Arousal_convert"].values < 0) & (features_list["Valence_convert"].values > 0)

            #decide weights
            hr_all = np.array(hr_all)
            hr_mean = np.mean(hr_all)
            w = np.ones(len(features_list.index)) * 0.5
            w[(hr_all > hr_mean) & a_h_v_n ] = 1.
            w[(hr_all < hr_mean) & a_l_v_n] = 1.
            w[(hr_all > hr_mean) & a_h_v_p] = 1.
            w[(hr_all < hr_mean) & a_l_v_p] = 1.


            features_list["weight"] = w

            all_features.append(features_list)
        except:
            print("Error" + subject)




df = pd.concat(all_features, ignore_index=True)

#resample
# df_postivie_ar = df[(df["Arousal_convert"].values == 1) & (df["Valence_convert"].values == 1) ].sample(frac=.7)
# df_postivie_val = df[(df["Arousal_convert"].values == 1) & (df["Valence_convert"].values == 0) ].sample(frac=1.)
#
# df_negative_ar = df[(df["Arousal_convert"].values == 0) & (df["Valence_convert"].values == 1) ].sample(frac=1.)
# df_negative_val = df[(df["Arousal_convert"].values == 0) & (df["Valence_convert"].values == 0) ].sample(frac=1.)
# #
# df = pd.concat([df_postivie_ar, df_postivie_val, df_negative_ar, df_negative_val])

#compute label porpotion

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
    training_data.to_csv(DATASET_PATH + "training_data_"+str(fold)+".csv", index=False)
    val_data.to_csv(DATASET_PATH + "validation_data_"+str(fold)+".csv", index=False)
    test_data.to_csv(DATASET_PATH + "test_data_"+str(fold)+".csv", index=False)
    fold += 1

# # Split to cross validation

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
#     training_data.to_csv(DATASET_PATH + "training_data_" + str(j) + ".csv", index=False)
#     val_data.to_csv(DATASET_PATH + "validation_data_" + str(j) + ".csv", index=False)


