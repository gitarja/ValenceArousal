from KnowledgeDistillation.Utils.DataGenerator import DataGenerator
from joblib import load
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_recall_fscore_support
path = "D:\\usr\\pras\\data\\EmotionTestVR\\"
training_file = "D:\\usr\\pras\\data\\EmotionTestVR\\training.csv"
testing_file = "D:\\usr\\pras\\data\\EmotionTestVR\\testing.csv"
ecg_length = 9000

generator = DataGenerator(path=path, training_list_file=training_file,
                          testing_list_file=testing_file, ecg_length=ecg_length)

train_inputs = generator.fetchData(training=True, split=True)

test_inputs = generator.fetchData(training=False, split=True)

scaler = load('..\\KnowledgeDistillation\\scaler.joblib')


train_features = []
train_y_ar = []
train_y_val = []
test_features = []
test_y_ar = []
test_y_val = []


for step, inputs in enumerate(train_inputs):
    X, _, y_val, y_ar = inputs
    train_features.append(scaler.transform(np.expand_dims(X, 0)))
    train_y_ar.append(y_ar)
    train_y_val.append(y_val)



for step, inputs in enumerate(test_inputs):
    X, _, y_val, y_ar = inputs
    test_features.append(scaler.transform(np.expand_dims(X, 0)))
    test_y_ar.append(y_ar)
    test_y_val.append(y_val)

train_features = np.concatenate(train_features)
test_features = np.concatenate(test_features)


forest_ar = ExtraTreesClassifier(n_estimators=50,
                                 random_state=0, max_depth=7, class_weight={0: 0.75, 1: 1.15})

forest_ar.fit(train_features, train_y_ar)

print(forest_ar.score(test_features, test_y_ar))
print(precision_recall_fscore_support(forest_ar.predict(test_features), test_y_ar, average='macro'))

forest_val = ExtraTreesClassifier(n_estimators=50,
                                 random_state=0, max_depth=7, class_weight={0: 0.75, 1: 1.15})

forest_val.fit(train_features, train_y_val)

print(forest_val.score(test_features, test_y_val))
print(precision_recall_fscore_support(forest_val.predict(test_features), test_y_val, average='macro'))


