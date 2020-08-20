import pandas as pd
import numpy as np
from Libs.Utils import valArLevelToLabels
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif


subjects = {"Okada", "Nishiwaki", "Komiya"}
features = []
y_ar = []
y_val = []

for s in subjects:
    path = "D:\\usr\\pras\\data\\EmotionTestVR\\"+s+"\\"
    eeg_path = path +"results\\EEG\\"
    gsr_path = path +"results\\GSR\\"
    resp_path = path +"results\\Resp\\"

    features_list = pd.read_csv(path+"features_list.csv")
    features_list["Valence"] = features_list["Valence"].apply(valArLevelToLabels)
    features_list["Arousal"] = features_list["Arousal"].apply(valArLevelToLabels)
    for i in range(len(features_list)):
        filename = features_list.iloc[i]["Idx"]
        eda_features = np.load(gsr_path+"eda_"+str(filename)+".npy")
        ppg_features = np.load(gsr_path + "ppg_" + str(filename) + ".npy")
        resp_features = np.load(resp_path + "resp_" + str(filename) + ".npy")
        eeg_features = np.load(eeg_path + "eeg_" + str(filename) + ".npy")

        features.append(np.concatenate([eda_features, ppg_features, resp_features, eeg_features]))

    y_ar.append(features_list["Arousal"].values)
    y_val.append(features_list["Valence"].values)


#concatenate features and normalize them
X = np.concatenate([features])
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

y_ar = np.concatenate(y_ar)
y_val = np.concatenate(y_val)



#features length
eda_length = eda_features.shape[0]
ppg_length = ppg_features.shape[0]
resp_length = resp_features.shape[0]
eeg_length = eeg_features.shape[0]

#Analyze arousal
# Build a forest and compute the impurity-based feature importances
X_ar_train, X_ar_test, y_ar_train, y_ar_test = train_test_split(X_norm, y_ar, test_size=0.3, random_state=42)
forest_ar = ExtraTreesClassifier(n_estimators=50,
                              random_state=0, max_depth=7, class_weight={0: 0.75, 1: 1.15})

forest_ar.fit(X_ar_train, y_ar_train)
print(forest_ar.score(X_ar_test, y_ar_test))

features_impt_ar = np.array([np.average(forest_ar.feature_importances_[0:eda_length]),
                          np.average(forest_ar.feature_importances_[eda_length:eda_length+ppg_length]),
                          np.average(forest_ar.feature_importances_[eda_length+ppg_length:eda_length+ppg_length+resp_length]),
                          np.average(forest_ar.feature_importances_[eda_length+ppg_length:eda_length+ppg_length+resp_length+eeg_length])])
print(features_impt_ar)
plt.bar(np.arange(len(features_impt_ar)), features_impt_ar)
plt.show()


#Analyze valence
X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_norm, y_val, test_size=0.3, random_state=42)
forest_val = ExtraTreesClassifier(n_estimators=50,
                              random_state=0, max_depth=7, class_weight={0: 0.75, 1: 1.15})

forest_val.fit(X_val_train, y_val_train)
print(forest_val.score(X_val_test, y_val_test))

features_impt_val = np.array([np.average(forest_val.feature_importances_[0:eda_length]),
                          np.average(forest_val.feature_importances_[eda_length:eda_length+ppg_length]),
                          np.average(forest_val.feature_importances_[eda_length+ppg_length:eda_length+ppg_length+resp_length]),
                          np.average(forest_val.feature_importances_[eda_length+ppg_length:eda_length+ppg_length+resp_length+eeg_length])])
print(features_impt_val)


plt.bar(np.arange(len(features_impt_val)), features_impt_val)
plt.show()

#Mutual informatiorn
mi_ar = mutual_info_classif(X_norm, y_ar)
mi_val = mutual_info_classif(X_norm, y_val)

avg_mi_ar = np.array([np.average(mi_ar[0:eda_length]),
                          np.average(mi_ar[eda_length:eda_length+ppg_length]),
                          np.average(mi_ar[eda_length+ppg_length:eda_length+ppg_length+resp_length]),
                          np.average(mi_ar[eda_length+ppg_length:eda_length+ppg_length+resp_length+eeg_length])])

avg_mi_val = np.array([np.average(mi_val[0:eda_length]),
                          np.average(mi_val[eda_length:eda_length+ppg_length]),
                          np.average(mi_val[eda_length+ppg_length:eda_length+ppg_length+resp_length]),
                          np.average(mi_val[eda_length+ppg_length:eda_length+ppg_length+resp_length+eeg_length])])


plt.bar(np.arange(len(avg_mi_ar)), avg_mi_ar)
plt.show()

plt.bar(np.arange(len(avg_mi_val)), avg_mi_val)
plt.show()
