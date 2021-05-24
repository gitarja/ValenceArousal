from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, \
    TRAINING_RESULTS_PATH, N_CLASS, ECG_N
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetch
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from Libs.Utils import calcAccuracyRegression
from sklearn import feature_selection
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis

fold = 4
training_data = DATASET_PATH + "\\stride=0.2\\training_data_" + str(fold) + ".csv"
validation_data = DATASET_PATH + "\\stride=0.2\\validation_data_" + str(fold) + ".csv"
testing_data = DATASET_PATH + "\\stride=0.2\\test_data_" + str(fold) + ".csv"

data_fetch = DataFetch(train_file=training_data, test_file=testing_data, validation_file=validation_data,
                       ECG_N=ECG_RAW_N, KD=False, teacher=True, ECG=False, high_only=False)

# extract train data
data = data_fetch.data_train
X_train = []
ar_train = []
val_train = []

for d in data:
    X_train.append(d[0])
    ar_train.append(d[2])
    val_train.append(d[3])

X_train = np.array(X_train)
ar_train = np.array(ar_train)
val_train = np.array(val_train)

# extract test data

data = data_fetch.data_test
X_test = []
ar_test = []
val_test = []

for d in data:
    X_test.append(d[0])
    ar_test.append(d[2])
    val_test.append(d[3])

X_test = np.array(X_test)
ar_test = np.array(ar_test)
val_test = np.array(val_test)

# features selection
nca_ar = NeighborhoodComponentsAnalysis(random_state=0)
nca_val = NeighborhoodComponentsAnalysis(random_state=0)
nca_ar.fit(X_train, ar_train)
nca_val.fit(X_train, val_train)

X_train_ar = nca_ar.transform(X_train)
X_train_val = nca_val.transform(X_train)
X_test_ar = nca_ar.transform(X_test)
X_test_val = nca_val.transform(X_test)

# X_train_ar = X_train
# X_train_val = X_train
# X_test_ar = X_test
# X_test_val = X_test

parameters = {"n_estimators": [50, 75, 100], "learning_rate": [0.1, 0.5, 1.]}
reg_ar = AdaBoostRegressor(ExtraTreeRegressor(max_depth=5, random_state=0), random_state=0)
reg_val = AdaBoostRegressor(ExtraTreeRegressor(max_depth=5, random_state=0), random_state=0)

clf_ar = GridSearchCV(reg_ar, parameters, verbose=2)
clf_val = GridSearchCV(reg_val, parameters, verbose=2)
clf_ar.fit(X_train_ar, ar_train)
clf_val.fit(X_train_val, val_train)

print(clf_ar.score(X_train_ar, ar_train))
print(clf_val.score(X_train_val, val_train))

print("--------------Test-------------------")
print(clf_ar.score(X_test_ar, ar_test))
print(clf_val.score(X_test_val, val_test))

print("-----------Summary------------------")
# print(classification_report(ar_test, clf_ar.predict(X_test)))
# print(classification_report(val_test, clf_val.predict(X_test)))

ar_predict = clf_ar.predict(X_test_ar)
val_predict = clf_val.predict(X_test_val)
th = 0.5
# ambigous

calcAccuracyRegression(ar_test, val_test, ar_predict, val_predict, mode="hard", th=th)
calcAccuracyRegression(ar_test, val_test, ar_predict, val_predict, mode="soft", th=th)
calcAccuracyRegression(ar_test, val_test, ar_predict, val_predict, mode="false", th=th)

# val positif
a_p = (ar_test > 0)
a_n = (ar_test < 0)
a_p_results = np.sum(ar_predict[a_p] > 0)
a_n_results = np.sum(ar_predict[a_n] <= 0)
print((a_p_results + a_n_results) / (np.sum(a_p) + np.sum(a_n)))
# val positif
v_p = (val_test > 0)
v_n = (val_test < 0)
v_p_results = np.sum(val_predict[v_p] > 0)
v_n_results = np.sum(val_predict[v_n] <= 0)
print((v_p_results + v_n_results) / (np.sum(v_p) + np.sum(v_n)))
