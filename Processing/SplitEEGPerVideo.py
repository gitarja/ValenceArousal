import pandas as pd
import os
from Libs.Utils import splitEEGPerVideo

date = '2020-10-30'
subject = 'E6'
path = 'G:\\usr\\nishihara\\data\\Yamaha-Experiment\\' + date + '\\EEG\\'
path_data = path + 'EEG_' + subject + '.csv'
path_gameresult = 'G:\\usr\\nishihara\\data\\Yamaha-Experiment\\' + date + '\\' + subject + '-' + date + '\\E6_M_2020_10_30_13_31_7_gameResults.csv'
eeg_data = pd.read_csv(path_data, header=None)
eeg_data.columns = ['Timestamp_Unix_CAL', 'ms', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'CH9', 'CH10',
                    'CH11', 'CH12', 'CH13', 'CH14', 'CH15', 'CH16', 'CH17', 'CH18', 'CH19']
gameresults = pd.read_csv(path_gameresult)
eeg_split = splitEEGPerVideo(eeg_data, gameresults)

path_newdir = path + 'EEG_' + subject + '\\'
os.makedirs(path_newdir, exist_ok=True)
for i, df in enumerate(eeg_split):
    path_split_data = path_newdir + 'eeg' + str(i) + '.csv'
    df.to_csv(path_split_data, index=False)

