import pandas as pd
import numpy as np
from Libs.Utils import valArLevelToLabels
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV
from StatisticalAnalysis.FeaturesImportance import FeaturesImportance
import glob

#init
features_importance = FeaturesImportance()

features = []
y_ar = []
y_val = []

data_path = "D:\\usr\\pras\\data\\YAMAHA\\Yamaha-Experiment (2020-10-26 - 2020-11-06)\\data\\*"
game_result = "\\*_gameResults.csv"
path_result = "results\\"

for folder in glob.glob(data_path):
    for subject in glob.glob(folder + "\\*-2020-*"):
        eeg_path = subject + "\\results\\EEG\\"
        eda_path = subject + "\\results\\eda\\"
        ppg_path = subject + "\\results\\ppg\\"
        resp_path = subject + "\\results\\Resp\\"
        ecg_path = subject + "\\results\\ECG\\"
        ecg_resp_path = subject + "\\results\\ECG_resp\\"

        features_list = pd.read_csv(subject + "\\features_list.csv")
        features_list["Valence"] = features_list["Valence"].apply(valArLevelToLabels)
        features_list["Arousal"] = features_list["Arousal"].apply(valArLevelToLabels)
        for i in range(len(features_list)):
            filename = features_list.iloc[i]["Idx"]
            eda_features = np.load(eda_path + "eda_" + str(filename) + ".npy")
            ppg_features = np.load(ppg_path + "ppg_" + str(filename) + ".npy")
            resp_features = np.load(resp_path + "resp_" + str(filename) + ".npy")
            eeg_features = np.load(eeg_path + "eeg_" + str(filename) + ".npy")
            ecg_features = np.load(ecg_path + "ecg_" + str(filename) + ".npy")
            ecg_resp_features = np.load(ecg_resp_path + "ecg_resp_" + str(filename) + ".npy")

            concat_features = np.concatenate(
                [eda_features, ppg_features, eeg_features])
            if np.sum(np.isinf(concat_features)) == 0 & np.sum(np.isnan(concat_features)) == 0:
                # print(eda_features.shape)
                features.append(concat_features)
                y_ar.append(features_list.iloc[i]["Arousal"])
                y_val.append(features_list.iloc[i]["Valence"])
            else:
                print(subject + "_" + str(i))

    # concatenate features and normalize them
X = np.concatenate([features])
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

y_ar = np.array(y_ar)
y_val = np.array(y_val)

# features length
eda_length = eda_features.shape[0]
ppg_length = ppg_features.shape[0]
resp_length = resp_features.shape[0]
eeg_length = eeg_features.shape[0]
ecg_length = ecg_features.shape[0]
ecg_resp_length = ecg_resp_features.shape[0]

print(eda_length)
print(ppg_length)
print(resp_length)
print(eeg_length)
print(ecg_length)
print(ecg_resp_length)

print("----------Class Propotion-------------------")
print("Arousal L-M: %f, Arousal M-H: %f" %(np.sum(y_ar==0) / len(y_ar), np.sum(y_ar==1) / len(y_ar)))
print("Valence L-M: %f, Valence M-H: %f" %(np.sum(y_ar==0) / len(y_val), np.sum(y_val==1) / len(y_val)))


# Mutual informatiorn
features_imp_ar = features_importance.getFeaturesImportance(X_norm, y_ar)
features_imp_val = features_importance.getFeaturesImportance(X_norm, y_val)

#anova
f_ar = features_imp_ar["anova_f"]
f_val = features_imp_val["anova_f"]
p_ar = features_imp_ar["anova_p"]
p_val = features_imp_val["anova_p"]
#mutual information
mi_ar = features_imp_ar["mi"]
mi_val =features_imp_val["mi"]

print("#------------------------------Start RF--------------------------------------------#")

# avg_mi_ar = np.array([np.average(mi_ar[0:eda_length]),
#                       np.average(mi_ar[eda_length:eda_length + ppg_length]),
#                       np.average(mi_ar[
#                                  eda_length + ppg_length:eda_length + ppg_length + resp_length]),
#                       np.average(mi_ar[
#                                  eda_length + ppg_length + resp_length:eda_length + ppg_length + resp_length + eeg_length]),
#                       np.average(mi_ar[
#                                  eda_length + ppg_length + resp_length + eeg_length:eda_length + ppg_length + resp_length + eeg_length + ecg_length]),
#                       np.average(mi_val[
#                                  eda_length + ppg_length + resp_length + eeg_length + ecg_length:eda_length + ppg_length + resp_length + eeg_length + ecg_length + ecg_resp_length])
#                       ])
#
# avg_mi_val = np.array([np.average(mi_val[0:eda_length]),
#                        np.average(mi_val[eda_length:eda_length + ppg_length]),
#                        np.average(mi_val[
#                                   eda_length + ppg_length:eda_length + ppg_length + resp_length]),
#                        np.average(mi_val[
#                                   eda_length + ppg_length + resp_length:eda_length + ppg_length + resp_length + eeg_length]),
#                        np.average(mi_val[
#                                   eda_length + ppg_length + resp_length + eeg_length:eda_length + ppg_length + resp_length + eeg_length + ecg_length]),
#                        np.average(mi_val[
#                                   eda_length + ppg_length + resp_length + eeg_length + ecg_length:eda_length + ppg_length + resp_length + eeg_length + ecg_length + ecg_resp_length])
#                        ])

# plt.bar(np.arange(len(avg_mi_ar)), avg_mi_ar)
# plt.xticks(np.arange(len(avg_mi_ar)), ["EDA", "PPG", "RESP", "EEG", "ECG", "ECG_RESP"])
# plt.show()
#
# plt.bar(np.arange(len(avg_mi_val)), avg_mi_val)
# plt.xticks(np.arange(len(avg_mi_val)), ["EDA", "PPG", "RESP", "EEG", "ECG", "ECG_RESP"])
# plt.show()


# Build a forest and compute the impurity-based feature importances
# Analyze arousal

#Forest params
parameters = {"n_estimators": [30, 50, 70], "max_depth": [3, 5, 7], "max_samples": [0.3, 0.5, 0.7],
              "max_features": [0.3, 0.5, 0.7, 1.], 'class_weight': [{0: 1.0 , 1: 1.15}, {0: 0.75, 1: 1.5}]}

X_ar_train, X_ar_test, y_ar_train, y_ar_test = train_test_split(X_norm, y_ar, test_size=0.3, random_state=42)
forest_ar = ExtraTreesClassifier(random_state=0)

clf_ar = GridSearchCV(forest_ar, parameters)
clf_ar.fit(X_ar_train, y_ar_train)
best_ar = clf_ar.best_estimator_

print(best_ar.score(X_ar_test, y_ar_test))
print(clf_ar.best_params_)

rf_ar = best_ar.feature_importances_

# features_impt_ar = np.array([np.average(best_ar.feature_importances_[0:eda_length]),
#                              np.average(best_ar.feature_importances_[eda_length:eda_length + ppg_length]),
#                              np.average(best_ar.feature_importances_[
#                                         eda_length + ppg_length:eda_length + ppg_length + resp_length]),
#                              np.average(best_ar.feature_importances_[
#                                         eda_length + ppg_length + resp_length:eda_length + ppg_length + resp_length + eeg_length]),
#                              np.average(best_ar.feature_importances_[
#                                         eda_length + ppg_length + resp_length + eeg_length:eda_length + ppg_length + resp_length + eeg_length + ecg_length]),
#                              np.average(best_ar.feature_importances_[
#                                         eda_length + ppg_length + resp_length + eeg_length + ecg_length:eda_length + ppg_length + resp_length + eeg_length + ecg_length + ecg_resp_length])
#                              ])
# print(features_impt_ar)
# plt.bar(np.arange(len(features_impt_ar)), features_impt_ar)
# plt.xticks(np.arange(len(features_impt_ar)), ["EDA", "PPG", "RESP", "EEG", "ECG", "ECG_RESP"])
# plt.show()



# Analyze valence
X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_norm, y_val, test_size=0.3, random_state=42)
forest_val = ExtraTreesClassifier(random_state=0)


clf_val = GridSearchCV(forest_ar, parameters)
clf_val.fit(X_val_train, y_val_train)
best_val = clf_val.best_estimator_

print(best_val.score(X_val_test, y_val_test))
print(clf_val.best_params_)

rf_val = best_val.feature_importances_

# features_impt_val = np.array([np.average(best_val.feature_importances_[0:eda_length]),
#                               np.average(best_val.feature_importances_[eda_length:eda_length + ppg_length]),
#                               np.average(best_val.feature_importances_[
#                                          eda_length + ppg_length:eda_length + ppg_length + resp_length]),
#                               np.average(best_val.feature_importances_[
#                                          eda_length + ppg_length + resp_length:eda_length + ppg_length + resp_length + eeg_length]),
#                               np.average(best_val.feature_importances_[
#                                          eda_length + ppg_length + resp_length + eeg_length:eda_length + ppg_length + resp_length + eeg_length + ecg_length]),
#                               np.average(best_val.feature_importances_[
#                                          eda_length + ppg_length + resp_length + eeg_length + ecg_length:eda_length + ppg_length + resp_length + eeg_length + ecg_length + ecg_resp_length])
#
# ])
# print(features_impt_val)
#
# plt.bar(np.arange(len(features_impt_val)), features_impt_val)
# plt.xticks(np.arange(len(features_impt_val)), ["EDA", "PPG", "RESP", "EEG", "ECG", "ECG_RESP"])
# plt.show()

#save results
np.savetxt(path_result+"f_ar.csv", f_ar, delimiter=",")
np.savetxt(path_result+"f_val.csv", f_val, delimiter=",")
np.savetxt(path_result+"p_ar.csv", p_val, delimiter=",")
np.savetxt(path_result+"p_val.csv", p_ar, delimiter=",")
np.savetxt(path_result+"mi_val.csv", mi_val, delimiter=",")
np.savetxt(path_result+"mi_ar.csv", mi_ar, delimiter=",")
np.savetxt(path_result+"rf_val.csv", rf_val, delimiter=",")
np.savetxt(path_result+"rf_ar.csv", rf_ar, delimiter=",")