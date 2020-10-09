import pandas as pd


subject = "Komiya"

path = "D:\\usr\\pras\\data\\EmotionTestVR\\"+subject+"\\"

eeg_features_list = pd.read_csv(path+"EEG_features_list.csv").set_index('Idx')
ecg_features_list = pd.read_csv(path+"ECG_features_list.csv").set_index('Idx')
GSR_features_list = pd.read_csv(path+"GSR_features_list.csv").set_index('Idx')
Resp_features_list = pd.read_csv(path+"Resp_features_list.csv").set_index('Idx')

features_list = eeg_features_list[(eeg_features_list["Status"]==1) & (GSR_features_list["Status"]==1) & (Resp_features_list["Status"]==1) & (ecg_features_list["Status"]==1)]

features_list.to_csv(path+"features_list.csv")
