import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from Libs.Utils import arToLabels, valToLabels, convertLabels, regressLabelsConv, convertLabelsReg
import glob
import numpy as np
from Conf.Settings import DATASET_PATH, STRIDE, ECG_PATH

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



for i in range(5):
    data_train = pd.read_csv(DATASET_PATH + "training_data_" + str(i))
    weight_train = df.loc[(df["Start"].isin(data_train["Start"].values)) & (df["End"].isin(data_train["End"].values)) & (df["Subject"].isin(data_train["Subject"].values)) ]
    data_train["weight"] = weight_train
    data_validation = pd.read_csv(DATASET_PATH + "validation_data_" + str(i))
    weight_validation = df.loc[
        (df["Start"].isin(data_validation["Start"].values)) & (df["End"].isin(data_validation["End"].values)) & (
            df["Subject"].isin(data_validation["Subject"].values))]
    data_validation["weight"] = weight_validation
    data_test = pd.read_csv(DATASET_PATH + "test_data_" + str(i))
    weight_test = df.loc[
        (df["Start"].isin(data_test["Start"].values)) & (df["End"].isin(data_test["End"].values)) & (
            df["Subject"].isin(data_test["Subject"].values))]
    data_test["weight"] = weight_test








