import pandas as pd

path = 'C:\\Users\\ShimaLab\\Documents\\nishihara\\data\\EmotionTestVR\\Komiyama\\'
ecg_path1 = path + '20200709_152213_165_HB_PW.csv'
ecg_path2 = path + '20200709_162217_514_HB_PW.csv'

ecg_data1 = pd.read_csv(ecg_path1)
ecg_data2 = pd.read_csv(ecg_path2)

ecg_concat = pd.concat([ecg_data1, ecg_data2])

ecg_concat.to_csv(path + 'Komiya_ECG.csv', index=False)
