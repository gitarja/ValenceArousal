import glob
from Conf.Settings import FS_EEG
from EEG.EEGFeatures import EEGFeatures
import pandas as pd

data_path = "D:\\usr\\pras\\data\\YAMAHA\\Yamaha-Experiment (2020-10-26 - 2020-11-06)\\data\\*"
eeg_file = "\\EEG\\"

eeg_features = EEGFeatures(fs=FS_EEG)

for folder in glob.glob(data_path):
    for subject in glob.glob(folder + "\\*-2020-10*"):
        try:
            for eeg_file_path in glob.glob(subject + eeg_file + "eeg*.csv"):
                eeg_data = pd.read_csv(eeg_file_path)
                time = (eeg_data.iloc[:, 0] + "." + eeg_data.iloc[:, 1].astype(str)).values
                eeg = eeg_data.iloc[:, 2:].values

                filtered_eeg = eeg_features.filterEEG(eeg)

                eeg_dict = {"time": time}
                for i in range(filtered_eeg.shape[1]):
                    eeg_dict["CH" + str(i)] = filtered_eeg[:, i]

                eeg_final = pd.DataFrame(eeg_dict)

                eeg_final.to_csv(eeg_file_path.replace("eeg", "filtered_eeg"), index=False)


        except:
            print(subject)
