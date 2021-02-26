import pandas as pd
from Libs.Utils import strTimeToUnixTime, unixTimeToStrTime
import numpy as np

FS = 1000
path = 'G:\\usr\\nishihara\\data\\Yamaha-Experiment\\2020-11-02\\E5-2020-11-02\\EEG\\'
path_data = path + 'EEG_E5_3.txt'
time_start = strTimeToUnixTime('2020/11/02 10:46:57')
time_end = strTimeToUnixTime('2020/11/02 10:50:11')
timestamp = np.arange(time_start, time_end, 1 / FS)
timestamp = pd.DataFrame([unixTimeToStrTime(t) for t in timestamp])

eeg_data = pd.read_table(path_data)
eeg = eeg_data.iloc[:len(timestamp), 1:21]
eeg = eeg.reset_index(drop=True)
eeg_organized = pd.concat([timestamp, eeg], axis=1)
eeg_organized.columns = ['Timestamp_Unix_CAL', 'ms', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'CH9',
                        'CH10', 'CH11', 'CH12', 'CH13', 'CH14', 'CH15', 'CH16', 'CH17', 'CH18', 'CH19']

eeg_organized.to_csv(path + 'eeg3_2.csv', index=False)
