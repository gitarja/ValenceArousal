import pandas as pd
from Libs.Utils import Utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from EEG.EEGFeatures import EEGFeatures
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

ut = Utils()
path = "F:\\data\\EEG+ECG\\Terui\\TeruiFiltered.csv"
data = pd.read_csv(path)
# convert timestamp to int
data["Time"] = data["Time"].apply(ut.timeToInt)

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
    ch1Features = timeExct.frequencyDomainFeatures(ut.butterBandpassFilter(g.CH1.values, lowcut=lc, highcut=hc, fs=fs), bands=bands, fs=fs)
    ch2Features = timeExct.frequencyDomainFeatures(ut.butterBandpassFilter(g.CH2.values, lowcut=lc, highcut=hc, fs=fs), bands=bands, fs=fs)
    ch3Features = timeExct.frequencyDomainFeatures(ut.butterBandpassFilter(g.CH3.values, lowcut=lc, highcut=hc, fs=fs), bands=bands, fs=fs)
    ch4Features = timeExct.frequencyDomainFeatures(ut.butterBandpassFilter(g.CH4.values, lowcut=lc, highcut=hc, fs=fs), bands=bands, fs=fs)
    ch5Features = timeExct.frequencyDomainFeatures(ut.butterBandpassFilter(g.CH5.values, lowcut=lc, highcut=hc, fs=fs), bands=bands, fs=fs)
    ch6Features = timeExct.frequencyDomainFeatures(ut.butterBandpassFilter(g.CH6.values, lowcut=lc, highcut=hc, fs=fs), bands=bands, fs=fs)
    ch7Features = timeExct.frequencyDomainFeatures(ut.butterBandpassFilter(g.CH7.values, lowcut=lc, highcut=hc, fs=fs), bands=bands, fs=fs)

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