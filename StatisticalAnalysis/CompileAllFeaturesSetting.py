import pandas as pd
from sklearn.model_selection import train_test_split
import glob


data_path = "D:\\usr\\pras\\data\\YAMAHA\\Yamaha-Experiment (2020-10-26 - 2020-11-06)\\data\\*"

for folder in glob.glob(data_path):
    for subject in glob.glob(folder + "\\*-2020-*"):
        try:
            eeg_features_list = pd.read_csv(subject+"\\EEG_features_list.csv").set_index('Idx')
            ecg_features_list = pd.read_csv(subject+"\\ECG_features_list.csv").set_index('Idx')
            GSR_features_list = pd.read_csv(subject+"\\GSR_features_list.csv").set_index('Idx')
            Resp_features_list = pd.read_csv(subject+"\\Resp_features_list.csv").set_index('Idx')

            features_list = ecg_features_list[(eeg_features_list["Status"]==1) & (GSR_features_list["Status"]==1) & (Resp_features_list["Status"]==1) & (ecg_features_list["Status"]==1)]
            features_list.to_csv(subject+"\\features_list.csv")
        except:
            print("Error: "+ subject)




