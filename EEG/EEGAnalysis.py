from EEG.EEGFeatures import EEGFeatures
import pandas as pd

from Libs.Utils import timeToInt
from Conf.Settings import FS_EEG
from EEG.SpaceLapFilter import SpaceLapFilter
import numpy as np

path = "D:\\usr\\pras\\data\\YAMAHA\\EEG\\"
# path_results = path + "results\\EEG\\"




format = '%H:%M:%S'
split_time = 45

eeg_features_list = pd.DataFrame(columns=["Idx", "Start", "End", "Valence", "Arousal", "Emotion", "Status", "Subject"])
idx = 0
eeg_filter = SpaceLapFilter()
eeg_features_exct = EEGFeatures(FS_EEG)
eeg_features_list = pd.DataFrame(columns=["Idx", "Start", "End", "Valence", "Arousal", "Emotion", "Status", "Subject"])
idx = 0
eeg_filter = SpaceLapFilter()
eeg_features_exct = EEGFeatures(FS_EEG)

eeg_data = pd.read_csv(path + "19_43_45_eegrawData.csv")

eeg_data.loc[:, "CH1":"CH19"] = eeg_filter.FilterEEG(eeg_data.loc[:, "CH1":"CH19"].values, mode=4)

eeg_filtered = eeg_data.loc[:, "CH1":"CH19"].values
# eeg_filtered = eeg_filter.FilterEEG(eeg, mode=4)
time_domain_features = eeg_features_exct.extractTimeDomainAll(eeg_filtered)
freq_domain_features = eeg_features_exct.extractFrequencyDomainAll(eeg_filtered)

if (time_domain_features.shape[0] != 0) & (freq_domain_features.shape[0] != 0):
    eeg_features = np.concatenate([time_domain_features, freq_domain_features])
    np.save(path + "eeg_" + str(idx) + ".npy", eeg_features)
    status = 1

eeg_features_list.to_csv(path + "EEG_features_list.csv")
