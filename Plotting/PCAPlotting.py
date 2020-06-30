import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
sns.axes_style("white")

path = "F:\\data\\EEG+ECG\\Terui\\ECG+EEGFreqFeatures.csv"
# path = "F:\\data\\EEG+ECG\\Pictures\\normalizedFeaturesECG.csv"
normalizedFeatures = pd.read_csv(path, encoding="UTF-8", sep=',', header=None).values

# PCA
pca = PCA(n_components=2)

pcaComponents = pca.fit_transform(normalizedFeatures)

# plotting
fig = plt.figure()
ax = fig.add_subplot(111)

ci = "y"
label = "V:L, A:L"
ax.scatter(pcaComponents[0:3, 0], pcaComponents[0:3, 1], c=ci, marker='o', label=label)

ci = "r"
label = "V:L, A:H"
ax.scatter(pcaComponents[3:6, 0], pcaComponents[3:6, 1], c=ci, marker='o', label=label)

ci = "b"
label = "V:H, A:H"
ax.scatter(pcaComponents[6:8, 0], pcaComponents[6:8, 1], c=ci, marker='o', label=label)

ci = "g"
label = "V:H, A:L"
ax.scatter(pcaComponents[8:14, 0], pcaComponents[8:14, 1], c=ci, marker='o', label=label)

for i in range(14):
    ax.annotate(str(i + 1), (pcaComponents[i, 0], pcaComponents[i, 1]))

ax.set_xlabel('1st Component')
ax.set_ylabel('2nd Component')
ax.legend()
ax.grid(True)
plt.show()
