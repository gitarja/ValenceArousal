import pandas as pd
import glob
import numpy as np

date = '2020-11-06'
subject = 'D4'
path = 'G:\\usr\\nishihara\\data\\Yamaha-Experiment\\' + date + '\\' + subject + '-' + date + '\\EEG\\'
path_datalist = glob.glob(path + '*.txt')
FS = 1000

for i, p in enumerate(path_datalist):
    eeg_data = pd.read_table(p, header=None)
    eeg_data = eeg_data.dropna(how='all', axis=1)
    eeg_data.iloc[:, 1] = np.arange(0, len(eeg_data), 1000 / FS, dtype=int)
    eeg_data.columns = ['Timestamp_Unix_CAL', 'ms', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'CH9',
                        'CH10', 'CH11', 'CH12', 'CH13', 'CH14', 'CH15', 'CH16', 'CH17', 'CH18', 'CH19']
    eeg_data.to_csv(path + 'eeg' + str(i) + '.csv', index=False)

