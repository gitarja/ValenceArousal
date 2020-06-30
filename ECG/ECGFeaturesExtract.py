from ECG.ECGFeatures import ECGFeatures
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from Libs.Utils import Utils
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')

ut = Utils()

path = "F:\\data\\EEG\\Pictures\\Mukaeda.csv"

data = pd.read_csv(path)
# start = "2020-04-29 15:40:21"
# end = "2020-04-29 15:54:56"
#
# data = data.loc[(data.timestamp >= start) & (data.timestamp <= end)]

# convert timestamp to int
data["timestamp"] = data["timestamp"].apply(ut.timeToInt)

# normalize the data
data["timestamp"] = data["timestamp"] - data.iloc[0].timestamp

# group for every 1 minute
groups = data.groupby(data["timestamp"].values // 13.)


# features extrator
featuresExct = ECGFeatures()
featuresEachMin = []
fs = 250
for i, g in groups:
    if (len(g.ecg.values) >= 3250):
        time_domain = featuresExct.extractTimeDomain(g.ecg.values, fs)
        freq_domain = featuresExct.extractFrequencyDomain(g.ecg.values, fs)
        nonlinear_domain = featuresExct.extractNonLinearDomain(g.ecg.values, fs)

        featuresEachMin.append(np.concatenate([time_domain, freq_domain, nonlinear_domain]))

#normalized features
featuresEachMin= np.where(np.isnan(featuresEachMin), 0, featuresEachMin)
featuresEachMin= np.where(np.isinf(featuresEachMin), 0, featuresEachMin)

normalizedFeatures = stats.zscore(featuresEachMin, 0)



np.savetxt('F:\\data\\EEG\\pictures\\normalizedFeaturesECG.csv', normalizedFeatures, delimiter=',')