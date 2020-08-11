from ECG.ECGFeatures import ECGFeatures
import pandas as pd
from Libs.Utils import timeToInt
from scipy import stats
import numpy as np
from Conf import Settings as set
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')

path = 'C:\\Users\\ShimaLab\\Documents\\nishihara\\data\\20200611\\ECG\\20200505_130107_379_HB_PW.csv'

data = pd.read_csv(path)
# start = "2020-04-29 15:40:21"
# end = "2020-04-29 15:54:56"
#
# data = data.loc[(data.timestamp >= start) & (data.timestamp <= end)]

# convert timestamp to int
data["timestamp"] = data["timestamp"].apply(timeToInt)

# normalize the data
data["timestamp"] = data["timestamp"] - data.iloc[0].timestamp

# group for every 1 minute
groups = data.groupby(data["timestamp"].values // 13.)


# features extrator
featuresExct = ECGFeatures(set.FS_ECG)
featuresEachMin = []
fs = 250
for i, g in groups:
    if (len(g.ecg.values) >= 3250):
        time_domain = featuresExct.extractTimeDomain(g.ecg.values)
        freq_domain = featuresExct.extractFrequencyDomain(g.ecg.values)
        nonlinear_domain = featuresExct.extractNonLinearDomain(g.ecg.values)

        featuresEachMin.append(np.concatenate([time_domain, freq_domain, nonlinear_domain]))


# normalized features
featuresEachMin = np.where(np.isnan(featuresEachMin), 0, featuresEachMin)
featuresEachMin = np.where(np.isinf(featuresEachMin), 0, featuresEachMin)

normalizedFeatures = stats.zscore(featuresEachMin, 0)


# plot
normalizedFeatures_T = normalizedFeatures.T
title = ['Mean NNI', 'Number of NNI', 'SDNN', 'Mean NNI difference', 'RMSSD', 'SDSD', 'Mean heart rate',
         'Std of the heart rate series', 'Normalized powers of LF', 'Normalized powers of HF', 'LF/HF ratio',
         'Sample entropy', 'Lyapunov exponent']
num_plot = 9
if normalizedFeatures_T.shape[0] % num_plot == 0:
    num_figure = normalizedFeatures_T.shape[0] // num_plot
else:
    num_figure = normalizedFeatures_T.shape[0] // num_plot + 1

for i in range(num_figure):
    plt.figure(figsize=(12, 9))
    for j in range(num_plot):
        if num_plot*i+j >= normalizedFeatures_T.shape[0]:
            break
        plt.subplot(3, 3, j + 1)
        plt.plot(normalizedFeatures_T[num_plot*i+j])
        plt.title(title[num_plot*i+j])
    plt.tight_layout()
plt.show()


np.savetxt('normalizedFeaturesECG.csv', normalizedFeatures, delimiter=',')

