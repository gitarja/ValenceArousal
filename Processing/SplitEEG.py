import pandas as pd
from Libs.Utils import splitEEGPerSubject

date = '2020-10-30'
path = 'G:\\usr\\nishihara\\data\\Yamaha-Experiment\\' + date + '\\EEG\\'
path_data = path + 'EEG_B2.txt'
path_timedata = 'G:\\usr\\nishihara\\data\\Yamaha-Experiment\\EEG_time.csv'
subjects = ['B2']
eeg_data = pd.read_table(path_data, header=None)
timedata = pd.read_csv(path_timedata, index_col=0)
eeg_split = splitEEGPerSubject(eeg_data, timedata, subjects)

for i, df in enumerate(eeg_split):
    path_split_data = path + '\\EEG_' + subjects[i] + '.csv'
    df.to_csv(path_split_data, header=False, index=False)
