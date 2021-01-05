import pandas as pd
import numpy as np
from Libs.Utils import arToLabels, valToLabels
from sklearn.preprocessing import StandardScaler
from StatisticalAnalysis.FeaturesImportance import FeaturesImportance
from Conf.Settings import ECG_PATH, STRIDE, DATASET_PATH_local, ROAD_TEST_PATH, SPLIT_TIME, FS_ECG, EXTENTION_TIME
from Libs.Utils import timeToInt
from ECG.ECGFeatures import ECGFeatures
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# init
features_importance = FeaturesImportance()

features = []
y_ar = []
y_val = []
ecg = []

game_result = "\\*_gameResults.csv"
path_result = "results\\"
subject_name = "E1-2020-11-03"
# Read VR Features
subject_path = DATASET_PATH_local + "2020-11-03\\"+ subject_name
features_list = pd.read_csv(subject_path+  "\\ECG_features_list_" + str(STRIDE) + ".csv")
features_list = features_list[features_list["Status"]==1]
features_list["Valence"] = features_list["Valence"].apply(valToLabels)
features_list["Arousal"] = features_list["Arousal"].apply(arToLabels)
for i in range(len(features_list)):
    filename = features_list.iloc[i]["Idx"]
    ecg_features = np.load(subject_path + ECG_PATH + "ecg_" + str(filename) + ".npy")
    if np.sum(np.isinf(ecg_features)) == 0 & np.sum(np.isnan(ecg_features)) == 0:
        features.append(ecg_features)
        y_ar.append(features_list.iloc[i]["Arousal"])
        y_val.append(features_list.iloc[i]["Valence"])
    else:
        print(subject_path + "_" + str(i))




#load road test features
subject_road_path = ROAD_TEST_PATH + "TS102 E1\\20201119_110408_783_HB_PW.csv"
ecg_road_data = pd.read_csv(subject_road_path)
# convert timestamp to int
ecg_road_data.loc[:, 'timestamp'] = ecg_road_data.loc[:, 'timestamp'].apply(timeToInt)
time_start = np.min(ecg_road_data["timestamp"].values)
time_end = np.max(ecg_road_data["timestamp"].values)
tdelta = time_end - time_start

#features extractor
featuresExct = ECGFeatures(FS_ECG)

#road test features
road_test_features = []
for j in np.arange(0, (tdelta // SPLIT_TIME), 0.2):
    end = time_end - ((j) * SPLIT_TIME)
    start = time_end - ((j+1) * SPLIT_TIME)- EXTENTION_TIME

    ecg = ecg_road_data[(ecg_road_data["timestamp"].values >= start) & (ecg_road_data["timestamp"].values <= end)]
    ecg_values = featuresExct.filterECG(ecg['ecg'].values)

    time_domain = featuresExct.extractTimeDomain(ecg_values)
    freq_domain = featuresExct.extractFrequencyDomain(ecg_values)
    nonlinear_domain = featuresExct.extractNonLinearDomain(ecg_values)
    if time_domain.shape[0] != 0 and freq_domain.shape[0] != 0 and nonlinear_domain.shape[0] != 0:
        concatenate_features = np.concatenate([time_domain, freq_domain, nonlinear_domain])
        road_test_features.append(concatenate_features)

road_test_features = np.concatenate([road_test_features])

# concatenate features and normalize them
X = np.concatenate([features, road_test_features])
scaler = StandardScaler()
scaler.fit(X)
X_norm = scaler.transform(X)
pca = PCA(n_components=3)
pca.fit(X_norm)
print(np.sum(pca.explained_variance_ratio_))

VR_norm_pca =  pca.transform(scaler.transform(features))
road_features_pca = pca.transform(scaler.transform(road_test_features))


#Compute distance
distance = euclidean_distances(VR_norm_pca, road_features_pca)
print(np.mean(distance))


#prepare labels
VR_LABELS = ["VR" for x in range(len(VR_norm_pca))]
ROAD_LABELS = ["ROAD" for x in range(len(road_features_pca))]


labels = np.concatenate([VR_LABELS, ROAD_LABELS])


#plot HR-Mean
coeff = np.concatenate([VR_norm_pca, road_features_pca], 0)
data = {"RR_Mean": X[:, 0] * 4, "RR_Std": X[:, 1] * 4, "HR_Mean": X[:, 6] / 4, "HR_Std": X[:, 7] / 4, "LF_HF": X[:, 10], "Lypanov": X[:, -1], "Experiment": labels}
features_data = pd.DataFrame(data)

plt.figure(1)
sns.boxplot(x="Experiment", y="RR_Mean",
                 data=features_data, palette="Set2").set_title('RR-Mean')
plt.figure(2)
sns.boxplot(x="Experiment", y="RR_Std",
                 data=features_data, palette="Set2").set_title('RR-STD')
plt.figure(3)
sns.boxplot(x="Experiment", y="HR_Mean",
                 data=features_data, palette="Set2").set_title('HR-Mean')
plt.figure(4)
sns.boxplot(x="Experiment", y="HR_Std",
                 data=features_data, palette="Set2").set_title('HR-STD')
plt.figure(5)
sns.boxplot(x="Experiment", y="LF_HF",
                 data=features_data, palette="Set2").set_title('LF/HF')
plt.figure(6)
sns.boxplot(x="Experiment", y="Lypanov",
                 data=features_data, palette="Set2").set_title('Lypanov Ex')

plt.show()

#
#
# coeff = np.concatenate([VR_norm_pca, road_features_pca], 0)
# data = {"coeff_x": coeff[:, 0], "coeff_y": coeff[:, 1], "coeff_z": coeff[:, 2], "Experiment": labels}
# pca_data = pd.DataFrame(data)




# #plotting
# fig = plt.figure()
# sns.color_palette("Set2")
#
# ax = fig.add_subplot(111, projection='3d')
# markers = ["o", "^"]
# groups =["VR", "ROAD"]
# ax.scatter(VR_norm_pca[:, 0], VR_norm_pca[:, 1],  VR_norm_pca[:, 2], marker=markers[0], label=groups[0])
# ax.scatter(road_features_pca[:, 0], road_features_pca[:, 1],  road_features_pca[:, 2], marker=markers[1], label=groups[1])
#
# ax.set_xlabel('Coeff 1')
# ax.set_ylabel('Coeff 2')
# ax.set_zlabel('Coeff 3')
# plt.legend()
# plt.show()
