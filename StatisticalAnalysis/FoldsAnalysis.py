import pandas as pd
import numpy as np
from Libs.Utils import valArLevelToLabels
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


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



    features_list_train = pd.read_csv(path_train+"features_list.csv")


    features_list_train["Valence"] = features_list_train["Valence"].apply(valArLevelToLabels)
    features_list_train["Arousal"] = features_list_train["Arousal"].apply(valArLevelToLabels)
    for i in range(len(features_list_train)):
        filename = features_list_train.iloc[i]["Idx"]
        eda_features = np.load(gsr_path_train+"eda_"+str(filename)+".npy")
        ppg_features = np.load(gsr_path_train + "ppg_" + str(filename) + ".npy")
        resp_features = np.load(resp_path_train + "resp_" + str(filename) + ".npy")
        eeg_features = np.load(eeg_path_train+ "eeg_" + str(filename) + ".npy")

        features_train.append(np.concatenate([eeg_features]))

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

    features_test.append(np.concatenate([eeg_features]))

#concatenate features and normalize them
scaler = StandardScaler()
X_train = np.concatenate([features_train])
X_test = np.concatenate([features_test])

X_norm_train = scaler.fit_transform(X_train)
X_norm_test = scaler.transform(X_test)

X = np.concatenate([X_norm_train, X_norm_test])



# label test
y_ar_test = features_list_test["Arousal"].values
y_val_test = features_list_test["Valence"].values


#concatenate label
y_ar = np.concatenate([y_ar_train, y_ar_test])
y_val = np.concatenate([y_val_train, y_val_test])

#hyperparameters
parameters = {"n_estimators": [50, 150, 200, 250], "max_depth": [3, 5, 7], 'class_weight':[{0:0.75, 1:1.15}, {0:0.75,1:1.5}]}

#Analyze arousal
# Build a forest and compute the impurity-based feature importances

X_ar_train, X_ar_test, y_ar_train, y_ar_test = train_test_split(X, y_ar, test_size=0.4, random_state=42)

forest_ar = ExtraTreesClassifier( random_state=0)
clf_ar = GridSearchCV(forest_ar, parameters)
clf_ar.fit(X_ar_train, y_ar_train)
print(clf_ar.score(X_ar_test, y_ar_test))
print(confusion_matrix(y_ar_test, clf_ar.predict(X_ar_test), normalize='all') * 100)


#Analyze valence
X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X, y_val, test_size=0.4, random_state=42)
forest_val = ExtraTreesClassifier(random_state=0)
clf_val = GridSearchCV(forest_val, parameters)
clf_val.fit(X_val_train, y_val_train)
print(clf_val.score(X_val_test, y_val_test))
print(confusion_matrix(y_val_test, clf_val.predict(X_val_test), normalize='all') * 100)
