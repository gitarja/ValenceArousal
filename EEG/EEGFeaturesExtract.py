import pandas as pd
from Libs.Utils import timeToInt, butterBandpassFilter
import matplotlib.pyplot as plt
from EEG.EEGFeatures import EEGFeatures
from Conf import Settings as set
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

path = "F:\\data\\EEG+ECG\\Terui\\TeruiFiltered.csv"
data = pd.read_csv(path)
# convert timestamp to int
data["Time"] = data["Time"].apply(timeToInt)

# normalize the data
data["Time"] = data["Time"] - data.loc[0].Time


# group for every 1 minute
groups = data.groupby(data["Time"].values // 60.)

# features extrator
timeExct = EEGFeatures()
featuresEachMin = []
lc = 4
hc = 55
fs = 1000
bands = [4,8,16,32]

for i, g in groups:
    ch1Features = timeExct.frequencyDomainFeatures(butterBandpassFilter(g.CH1.values, lowcut=lc, highcut=hc, fs=fs), bands=bands)
    ch2Features = timeExct.frequencyDomainFeatures(butterBandpassFilter(g.CH2.values, lowcut=lc, highcut=hc, fs=fs), bands=bands)
    ch3Features = timeExct.frequencyDomainFeatures(butterBandpassFilter(g.CH3.values, lowcut=lc, highcut=hc, fs=fs), bands=bands)
    ch4Features = timeExct.frequencyDomainFeatures(butterBandpassFilter(g.CH4.values, lowcut=lc, highcut=hc, fs=fs), bands=bands)
    ch5Features = timeExct.frequencyDomainFeatures(butterBandpassFilter(g.CH5.values, lowcut=lc, highcut=hc, fs=fs), bands=bands)
    ch6Features = timeExct.frequencyDomainFeatures(butterBandpassFilter(g.CH6.values, lowcut=lc, highcut=hc, fs=fs), bands=bands)
    ch7Features = timeExct.frequencyDomainFeatures(butterBandpassFilter(g.CH7.values, lowcut=lc, highcut=hc, fs=fs), bands=bands)

    featuresEachMin.append(
        np.concatenate([ch1Features, ch2Features, ch3Features, ch4Features, ch5Features, ch6Features, ch7Features]))

#normalized features
normalizedFeatures = stats.zscore(featuresEachMin, 0)

#PCA
pca = PCA(n_components=2)
pcaComponents = pca.fit_transform(normalizedFeatures)

#plotting
fig = plt.figure()
ax = fig.add_subplot(111)
colors = ['b', 'g', 'r', 'y', 'k', 'm', 'k']
for i in range(10):

    ax.scatter(pcaComponents[i, 0], pcaComponents[i, 1], c=colors[i//2], marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

plt.show()

np.savetxt('F:\\data\\EEG+ECG\\Terui\\normalizedFeaturesEEGFreq_2.csv', normalizedFeatures, delimiter=',')