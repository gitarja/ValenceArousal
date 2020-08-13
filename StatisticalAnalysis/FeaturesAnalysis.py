import pandas as pd
import numpy as np
from Libs.Utils import valArLevelToLabels
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

path = "D:\\usr\\pras\\data\\EmotionTestVR\\Komiya\\"
eeg_path = path +"results\\EEG\\"
gsr_path = path +"results\\GSR\\"
resp_path = path +"results\\Resp\\"

features_list = pd.read_csv(path+"features_list.csv")

features = []
features_list["Valence"] = features_list["Valence"].apply(valArLevelToLabels)
features_list["Arousal"] = features_list["Arousal"].apply(valArLevelToLabels)
for i in range(len(features_list)):
    filename = features_list.iloc[i]["Idx"]
    eda_features = np.load(gsr_path+"eda_"+str(filename)+".npy")
    ppg_features = np.load(gsr_path + "ppg_" + str(filename) + ".npy")
    resp_features = np.load(resp_path + "resp_" + str(filename) + ".npy")
    eeg_features = np.load(eeg_path + "eeg_" + str(filename) + ".npy")

    features.append(np.concatenate([eda_features, ppg_features, resp_features, eeg_features]))

#concatenate features and normalize them
X = np.concatenate([features])
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)


y_ar = features_list["Arousal"].values
y_val = features_list["Valence"].values


#features length
eda_length = eda_features.shape[0]
ppg_length = ppg_features.shape[0]
resp_length = resp_features.shape[0]
eeg_length = eeg_features.shape[0]

#Analyze arousal
# Build a forest and compute the impurity-based feature importances
forest_ar = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

X_ar_train, X_ar_test, y_ar_train, y_ar_test = train_test_split(X_norm, y_ar, test_size=0.4, random_state=42)
forest_ar.fit(X_ar_train, y_ar_train)
print(forest_ar.score(X_ar_test, y_ar_test))

features_impt_ar = np.array([np.sum(forest_ar.feature_importances_[0:eda_length]),
                          np.sum(forest_ar.feature_importances_[eda_length:eda_length+ppg_length]),
                          np.sum(forest_ar.feature_importances_[eda_length+ppg_length:eda_length+ppg_length+resp_length]),
                          np.sum(forest_ar.feature_importances_[eda_length+ppg_length:eda_length+ppg_length+resp_length+eeg_length])])
print(features_impt_ar)
plt.bar(np.arange(len(features_impt_ar)), features_impt_ar)

plt.show()


#Analyze valence
forest_val = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_norm, y_val, test_size=0.4, random_state=42)
forest_val.fit(X_val_train, y_val_train)
print(forest_val.score(X_val_test, y_val_test))

features_impt_val = np.array([np.sum(forest_val.feature_importances_[0:eda_length]),
                          np.sum(forest_val.feature_importances_[eda_length:eda_length+ppg_length]),
                          np.sum(forest_val.feature_importances_[eda_length+ppg_length:eda_length+ppg_length+resp_length]),
                          np.sum(forest_val.feature_importances_[eda_length+ppg_length:eda_length+ppg_length+resp_length+eeg_length])])
print(features_impt_val)
plt.bar(np.arange(len(features_impt_val)), features_impt_val)


plt.show()
