import pandas as pd
from Libs.Utils import strTimeToUnixTime

subject = 'A6-2020-10-27'
date = '2020-10-27'
path = 'G:\\usr\\nishihara\\data\\Yamaha-Experiment\\' + date + '\\' + subject + '\\'
gsr_data = pd.read_csv(path + 'Resp\\EmotionTest_Session3_RESP_Calibrated_SD.csv', header=[0, 1])
gsr_data = gsr_data.dropna(how='all', axis=1)
gsr_data = gsr_data.rename(columns={'RESP_Timestamp_FormattedUnix_CAL': 'RESP_Timestamp_Unix_CAL',
                                    'yyyy/mm/dd hh:mm:ss.000': 'ms'})
gsr_data.iloc[:, 0] = [strTimeToUnixTime(t, form='%Y/%m/%d %H:%M:%S.%f') * 1000
                       for t in gsr_data.iloc[:, 0]]

gsr_data.to_csv(path + 'Resp\\' + subject + '_Resp.csv', index=False)
