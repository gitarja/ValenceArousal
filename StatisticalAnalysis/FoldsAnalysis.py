import pandas as pd
import numpy as np
from Libs.Utils import valArLevelToLabels
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import classification_report


#training data
subjects = {"Okada", "Nishiwaki"}
features_train = []
y_ar_train = []
y_val_train = []
for s in subjects:
    path_train = "D:\\usr\\pras\\data\\EmotionTestVR\\"+s+"\\"
    eeg_path_train = path_train +"results\\EEG\\"
    gsr_path_train = path_train +"results\\GSR\\"
    resp_path_train = path_train +"results\\Resp\\"
    ecg_path_train = path_train + "results\\ECG\\"



    features_list_train = pd.read_csv(path_train+"features_list.csv")


    features_list_train["Valence"] = features_list_train["Valence"].apply(valArLevelToLabels)
    features_list_train["Arousal"] = features_list_train["Arousal"].apply(valArLevelToLabels)
    for i in range(len(features_list_train)):
        filename = features_list_train.iloc[i]["Idx"]
        eda_features = np.load(gsr_path_train+"eda_"+str(filename)+".npy")
        ppg_features = np.load(gsr_path_train + "ppg_" + str(filename) + ".npy")
        resp_features = np.load(resp_path_train + "resp_" + str(filename) + ".npy")
        eeg_features = np.load(eeg_path_train+ "eeg_" + str(filename) + ".npy")
        ecg_features = np.load(ecg_path_train + "ecg_" + str(filename) + ".npy")

        features_train.append(np.concatenate([eda_features,ppg_features, resp_features, eeg_features, ecg_features]))

    # label train
    y_ar_train.append(features_list_train["Arousal"].values)
    y_val_train.append(features_list_train["Valence"].values)

y_ar_train = np.concatenate(y_ar_train)
y_val_train = np.concatenate(y_val_train)




#testing data
path_test = "D:\\usr\\pras\\data\\EmotionTestVR\\Komiya\\"
eeg_path_test  = path_test +"results\\EEG\\"
gsr_path_test  = path_test +"results\\GSR\\"
resp_path_test  = path_test +"results\\Resp\\"
ecg_path_test = path_test +"results\\ECG\\"



features_list_test = pd.read_csv(path_test+"features_list.csv")

features_test = []
features_list_test["Valence"] = features_list_test["Valence"].apply(valArLevelToLabels)
features_list_test["Arousal"] = features_list_test["Arousal"].apply(valArLevelToLabels)
for i in range(len(features_list_test)):
    filename = features_list_test.iloc[i]["Idx"]
    eda_features = np.load(gsr_path_test +"eda_"+str(filename)+".npy")
    ppg_features = np.load(gsr_path_test  + "ppg_" + str(filename) + ".npy")
    resp_features = np.load(resp_path_test  + "resp_" + str(filename) + ".npy")
    eeg_features = np.load(eeg_path_test + "eeg_" + str(filename) + ".npy")
    ecg_features = np.load(ecg_path_test + "ecg_" + str(filename) + ".npy")

    features_test.append(np.concatenate([eda_features,ppg_features, resp_features, eeg_features, ecg_features]))

#concatenate features and normalize them
scaler = StandardScaler()
X = np.concatenate([features_train, features_test])

X = scaler.fit_transform(X)





# label test
y_ar_test = features_list_test["Arousal"].values
y_val_test = features_list_test["Valence"].values


#concatenate label
y_ar = np.concatenate([y_ar_train, y_ar_test])
y_val = np.concatenate([y_val_train, y_val_test])

#hyperparameters
# parameters = {"n_estimators": [25, 50, 75, 100], "max_depth": [2, 3, 5], 'class_weight':[{0:0.75, 1:1.}, {0:1. ,1:1.}]}
parameters = {"n_estimators": [25, 50, 75, 100]}

#Analyze arousal
# Build a forest and compute the impurity-based feature importances
ar_predicts = []
ar_truth = []
kf = StratifiedKFold(n_splits=5, shuffle=True)
target_names = ["L-M", "M-H"]
for train_index, test_index in kf.split(X, y_ar):
    X_train = X[train_index]
    X_test = X[test_index]


    y_ar_train = y_ar[train_index]
    y_ar_test = y_ar[test_index]

    forest_ar = AdaBoostClassifier(base_estimator=ExtraTreeClassifier(max_depth=3, max_features=0.5), random_state=0)
    clf_ar = GridSearchCV(forest_ar, parameters)
    clf_ar.fit(X_train, y_ar_train)
    print(clf_ar.best_params_)
    print(clf_ar.score(X_test, y_ar_test))
    ar_predicts.append(clf_ar.predict(X_test))
    ar_truth.append(y_ar_test)

print(classification_report(np.concatenate(ar_truth), np.concatenate(ar_predicts), target_names=target_names))
print("-----------------------------------------------------------------------------------------")
#Analyze valence
val_predicts = []
val_truth = []
for train_index, test_index in kf.split(X, y_val):

    X_train = X[train_index]
    X_test = X[test_index]


    y_val_train = y_val[train_index]
    y_val_test = y_val[test_index]


    forest_val = AdaBoostClassifier(base_estimator=ExtraTreeClassifier(max_depth=3, max_features=0.5), random_state=0)
    clf_val = GridSearchCV(forest_val, parameters)
    clf_val.fit(X_train, y_val_train)
    print(clf_val.best_params_)
    print(clf_val.score(X_test, y_val_test))
    val_predicts.append(clf_val.predict(X_test))
    val_truth.append(y_val_test)

print(classification_report(np.concatenate(val_truth), np.concatenate(val_predicts), target_names=target_names))
