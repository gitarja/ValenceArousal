import pandas as pd
import numpy as np
import glob
import os
from Conf.Settings import DATASET_PATH, STRIDE
from Libs.Utils import regressLabelsConv

TH = 0.5
distribution = pd.DataFrame(np.zeros(shape=(1, 7)), columns=["A1", "A2", "A3", "B1", "B2", "B3", "C"], dtype="int64")

for folder in glob.glob(DATASET_PATH + "2020-*\\"):
    for subject in glob.glob(folder + "*-2020-*\\"):
        features_list = pd.read_csv(subject + "features_list_" + str(STRIDE) + ".csv")
        valence = features_list["Valence"].apply(regressLabelsConv).values
        arousal = features_list["Arousal"].apply(regressLabelsConv).values

        distribution["A1"] += np.count_nonzero((valence < -TH) & (arousal > TH))
        distribution["A2"] += np.count_nonzero((valence < -TH) & ((arousal >= -TH) & (arousal <= TH)))
        distribution["A3"] += np.count_nonzero((valence < -TH) & (arousal < -TH))
        distribution["B1"] += np.count_nonzero((valence > TH) & (arousal > TH))
        distribution["B2"] += np.count_nonzero((valence > TH) & ((arousal >= -TH) & (arousal <= TH)))
        distribution["B3"] += np.count_nonzero((valence > TH) & (arousal < -TH))
        distribution["C"] += np.count_nonzero((valence >= -TH) & (valence <= TH))

print(distribution.sum(axis=1).values[0])
distribution /= distribution.sum(axis=1).values[0]
result_path = "D:\\usr\\nishihara\\result\\LabelDistribution\\"
os.makedirs(result_path, exist_ok=True)
distribution.to_csv(result_path + "LabelDistribution.csv", index=False)

