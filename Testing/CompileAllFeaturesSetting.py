import pandas as pd

path = "D:\\usr\\pras\\data\\EmotionTestVR\\Komiya\\"

eeg_features_list = pd.read_csv(path+"EEG_features_list.csv").set_index('Idx')
GSR_features_list = pd.read_csv(path+"GSR_features_list.csv").set_index('Idx')
Resp_features_list = pd.read_csv(path+"Resp_features_list.csv").set_index('Idx')

features_list = eeg_features_list[(eeg_features_list["Status"]==1) & (GSR_features_list["Status"]==1) & (GSR_features_list["Status"]==1)]

features_list.to_csv(path+"features_list.csv")
