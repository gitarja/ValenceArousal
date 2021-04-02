from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, TRAINING_RESULTS_PATH, N_CLASS, ECG_N
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetch
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import feature_selection
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.metrics import classification_report
import pandas as pd
fold = 5
training_data = DATASET_PATH + "\\stride=0.2\\training_data_" + str(fold) + ".csv"
validation_data = DATASET_PATH + "\\stride=0.2\\validation_data_" + str(fold) + ".csv"
testing_data = DATASET_PATH + "\\stride=0.2\\test_data_" + str(fold) + ".csv"

data_fetch = DataFetch(train_file=training_data, test_file=testing_data, validation_file=validation_data,
                       ECG_N=ECG_RAW_N, KD=True, teacher=False, ECG=False, high_only=False)

#extract train data
data = data_fetch.data_train
X_train = []
ar_train = []
val_train = []

for d in data:
    X_train.append(d[5])
    ar_train.append(d[2])
    val_train.append(d[3])

X_train = np.array(X_train)
ar_train = np.array(ar_train)
val_train = np.array(val_train)

#extract test data

data = data_fetch.data_test
X_test = []
ar_test = []
val_test = []

for d in data:
    X_test.append(d[5])
    ar_test.append(d[2])
    val_test.append(d[3])

X_test = np.array(X_test)
ar_test = np.array(ar_test)
val_test = np.array(val_test)
parameters = {"n_estimators": [150], "learning_rate": [0.95]}
reg_ar = AdaBoostClassifier(ExtraTreeClassifier(max_depth=15,  random_state=0), random_state=0)
reg_val = AdaBoostClassifier(ExtraTreeClassifier(max_depth=15,  random_state=0), random_state=0)

#features selectrion
# select_ar = feature_selection.SelectKBest(feature_selection.mutual_info_regression, k=2)
# select_ar.fit(X_train, ar_train)
# select_val =  feature_selection.SelectKBest(feature_selection.mutual_info_regression, k=2)
# select_val.fit(X_train, val_train)

# X_train_ar = select_ar.transform(X_train)
# X_train_val = select_val.transform(X_train)
# X_test_ar = select_ar.transform(X_test)
# X_test_val = select_val.transform(X_test)

X_train_ar = X_train
X_train_val = X_train
X_test_ar = X_test
X_test_val = X_test

clf_ar = GridSearchCV(reg_ar, parameters)
clf_val = GridSearchCV(reg_val, parameters)
clf_ar.fit(X_train_ar, ar_train)
clf_val.fit(X_train_val, val_train)

print(clf_ar.score(X_train_ar, ar_train))
print(clf_val.score(X_train_val, val_train))

print("--------------Test-------------------")
print(clf_ar.score(X_test_ar, ar_test))
print(clf_val.score(X_test_val, val_test))


print ("-----------Summary------------------")
# print(classification_report(ar_test, clf_ar.predict(X_test)))
# print(classification_report(val_test, clf_val.predict(X_test)))

ar_results = np.array([clf_ar.predict(X_test_ar), ar_test]).transpose()
val_results = np.array([clf_val.predict(X_test_val), val_test]).transpose()
th = 0.5
# ambigous

# th = 0.5 - th
# ar_results[:, 0] = ar_results[:, 0] + np.sign(ar_results[:, 0])
# val_results[:, 0] = val_results[:, 0] + np.sign(val_results[:, 0])
# ar positif and val positif
ar_p_v_p = (ar_results[:, 1] > th) & (val_results[:, 1] > th)
ar_p_v_p_results = np.average((ar_results[ar_p_v_p, 0] > 0) & (val_results[ar_p_v_p, 0] > 0))
# print("AR-pos and Val-pos: " + str(ar_p_v_p_results))
# ar positif and val negatif
ar_p_v_n = (ar_results[:, 1] > th) & (val_results[:, 1] < -th)
ar_p_v_n_results = np.average((ar_results[ar_p_v_n, 0] > 0) & (val_results[ar_p_v_n, 0] < -0))
# print("AR-pos and Val-neg: " + str(ar_p_v_n_results))
# ar negatif and val positif
ar_n_v_p = (ar_results[:, 1] < -th) & (val_results[:, 1] > th)
ar_n_v_p_results = np.average((ar_results[ar_n_v_p, 0] < -0) & (val_results[ar_n_v_p, 0] > 0))
# print("AR-neg and Val-pos: " + str(ar_n_v_p_results))
# ar negatif and val negatif
ar_n_v_n = (ar_results[:, 1] < -th) & (val_results[:, 1] < -th)
ar_n_v_n_results = np.average((ar_results[ar_n_v_n, 0] < -0) & (val_results[ar_n_v_n, 0] < -0))
# print("AR-neg and Val-neg: " + str(ar_n_v_n_results))

# val positif
a_p = (ar_results[:, 1] > 0)
a_n = (ar_results[:, 1] < 0)
a_p_results = np.sum(ar_results[a_p, 0] > 0)
a_n_results = np.sum(ar_results[a_n, 0] <= 0)
# print((a_p_results + a_n_results) / (np.sum(a_p) + np.sum(a_n)))
# val positif
v_p = (val_results[:, 1] > 0)
v_n = (val_results[:, 1] < 0)
v_p_results = np.sum(val_results[v_p, 0] > 0)
v_n_results = np.sum(val_results[v_n, 0] <= 0)
# print((v_p_results + v_n_results) / (np.sum(v_p) + np.sum(v_n)))

print(str(ar_p_v_p_results) + "," + str(ar_p_v_n_results) + "," + str(ar_n_v_p_results) + "," + str(ar_n_v_n_results))