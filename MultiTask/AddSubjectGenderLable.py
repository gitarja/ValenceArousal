import numpy as np
import pandas as pd
from Conf.Settings import DATASET_PATH
import os

path = DATASET_PATH + "stride=0.2\\"
path_result = DATASET_PATH + "stride=0.2_multitask\\"
os.makedirs(path_result, exist_ok=True)
for fold in range(1, 6):
    df_train = pd.read_csv(path + "training_data_" + str(fold) + ".csv")
    df_val = pd.read_csv(path + "validation_data_" + str(fold) + ".csv")
    df_test = pd.read_csv(path + "test_data_" + str(fold) + ".csv")

    subject_labels_train, _ = pd.factorize(df_train["Subject"], sort=True)
    subject_labels_val, _ = pd.factorize(df_val["Subject"], sort=True)
    subject_labels_test, _ = pd.factorize(df_test["Subject"], sort=True)

    df_train["Subject_label"] = subject_labels_train
    df_val["Subject_label"] = subject_labels_val
    df_test["Subject_label"] = subject_labels_test

    female_list = ["C1-2020-11-04", "C2-2020-11-04", "C3-2020-10-29", "C4-2020-10-29", "E2-2020-10-29", "E3-2020-11-04", "E4-2020-11-04"]
    gender_label_train = np.zeros_like(subject_labels_train)
    gender_label_val = np.zeros_like(subject_labels_val)
    gender_label_test = np.zeros_like(subject_labels_test)

    for female in female_list:
        gender_label_train[df_train["Subject"].values == female] = 1
        gender_label_val[df_val["Subject"].values == female] = 1
        gender_label_test[df_test["Subject"].values == female] = 1

    df_train["Gender_label"] = gender_label_train
    df_val["Gender_label"] = gender_label_val
    df_test["Gender_label"] = gender_label_test

    df_train.to_csv(path_result + "training_data_" + str(fold) + ".csv", index=False)
    df_val.to_csv(path_result + "validation_data_" + str(fold) + ".csv", index=False)
    df_test.to_csv(path_result + "test_data_" + str(fold) + ".csv", index=False)

