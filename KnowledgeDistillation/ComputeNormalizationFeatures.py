from Libs.DataGenerator import DataGenerator
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit




path = "D:\\usr\\pras\\data\\EmotionTestVR\\"
training_file = "D:\\usr\\pras\\data\\EmotionTestVR\\training.csv"
testing_file = "D:\\usr\\pras\\data\\EmotionTestVR\\testing.csv"
feature_list_file = "D:\\usr\\pras\\data\\EmotionTestVR\\features_list.csv"
ecg_length = 11000
generator = DataGenerator(path=path, training_list_file=feature_list_file,
                          testing_list_file=testing_file, ecg_length=ecg_length)

X = generator.fetchData(training=True, split=False)
features = []
Y = []
for step, inputs in enumerate(X):
    X, _, y = inputs
    features.append(np.expand_dims(X, 0))
    Y.append(y)

features = np.concatenate(features)

#Compute z normalization
# scaler = StandardScaler()
# scaler.fit(features)
# dump(scaler, 'scaler.joblib')


#Data TrainTest split

dataset = pd.read_csv(feature_list_file)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
for train_index, test_index in sss.split(features, Y):
    data_train = dataset.iloc[train_index]
    data_test = dataset.iloc[test_index]

    #save data
    data_train.to_csv(path + "training.csv")
    data_test.to_csv(path + "testing.csv")
