from ECG.ECGFeatures import ECGFeatures
import pandas as pd
from Libs.Utils import timeToInt
from scipy import stats
import numpy as np
from Conf import Settings as set
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')

path = 'C:\\Users\\ShimaLab\\Documents\\nishihara\\data\\20200709\\Komiyama\\'
path_ECGdata = path + '20200709_152213_165_HB_PW.csv'
path_EmotionTest = path + 'Komiya_M_2020_7_9_15_22_44_gameResults.csv'
path_results = path + 'results\\ECG\\'

data = pd.read_csv(path_ECGdata)
data_EmotionTest = pd.read_csv(path_EmotionTest)
# start = "2020-04-29 15:40:21"
# end = "2020-04-29 15:54:56"
#
# data = data.loc[(data.timestamp >= start) & (data.timestamp <= end)]

# convert timestamp to int
data.loc[:, 'timestamp'] = data.loc[:, 'timestamp'].apply(timeToInt)
data_EmotionTest.loc[:, 'Time_Start'] = data_EmotionTest.loc[:, 'Time_Start'].apply(timeToInt)
data_EmotionTest.loc[:, 'Time_End'] = data_EmotionTest.loc[:, 'Time_End'].apply(timeToInt)

data_valid = []
for i in list(data_EmotionTest.index):
    tmp = data.loc[(data.loc[:, 'timestamp'] >= data_EmotionTest.at[i, 'Time_Start']) & (data.loc[:, 'timestamp'] <= data_EmotionTest.at[i, 'Time_End'])]
    if not tmp.empty:
        data_valid.append(tmp)

# features extractor
featuresExct = ECGFeatures(set.FS_ECG)
time = []
emotionTestResult = pd.DataFrame(columns=["Idx", "Start", "End", "Valence", "Arousal", "Emotion", "Status"])
ecgFeatures = []
windowsize = 45
slide = 18
idx = 0
itr = 0

for df in data_valid:
    # normalize time data
    normalizedTime = df.loc[:, 'timestamp'].values - df.iat[0, 0]

    featuresEachMin = []
    t0 = 0
    tf = windowsize
    idx0 = 0
    idxf = np.where(np.array([int(n) for n in normalizedTime]) == tf)[0][0]
    while tf <= normalizedTime[-1]:
        time_start = df.iat[idx0, 0]
        time_end = df.iat[idxf, 0]
        time_domain = featuresExct.extractTimeDomain(df.loc[:, 'ecg'].values[idx0:idxf])
        freq_domain = featuresExct.extractFrequencyDomain(df.loc[:, 'ecg'].values[idx0:idxf])
        nonlinear_domain = featuresExct.extractNonLinearDomain(df.loc[:, 'ecg'].values[idx0:idxf])
        if time_domain.shape[0] != 0 and freq_domain.shape[0] != 0 and nonlinear_domain.shape[0] != 0:
            featuresEachMin.append(np.concatenate([time_domain, freq_domain, nonlinear_domain]))
            time.append(np.average(normalizedTime[idx0:idxf]))
            t0 += slide
            tf += slide
            if tf > normalizedTime[-1]:
                break
            else:
                idx0 = np.where(np.array([int(n) for n in normalizedTime]) == t0)[0][0]
                idxf = np.where(np.array([int(n) for n in normalizedTime]) == tf)[0][0]
            np.save(path_results + "ecg_" + str(idx) + ".npy", featuresEachMin)
            status = 1
        else:
            status = 0

        emotionTestResult = emotionTestResult.append(
            {'Idx': idx, 'Start': time_start, 'End': time_end,
             "Valence": data_EmotionTest.iat[itr, 4], "Arousal": data_EmotionTest.iat[itr, 5],
             "Emotion": data_EmotionTest.iat[itr, 6], "Status": status}, ignore_index=True)
        idx += 1

    featuresEachMin = np.where(np.isnan(featuresEachMin), 0, featuresEachMin)
    featuresEachMin = np.where(np.isinf(featuresEachMin), 0, featuresEachMin)
    if itr == 0:
        ecgFeatures = featuresEachMin
    else:
        ecgFeatures = np.concatenate([ecgFeatures, featuresEachMin])

    itr += 1

# save to csv
emotionTestResult.to_csv(path + 'Komiyama_ECG.csv')


# plot
title = ['Mean NNI', 'Number of NNI', 'SDNN', 'Mean NNI difference', 'RMSSD', 'SDSD', 'Mean heart rate',
         'Std of the heart rate series', 'Normalized powers of LF', 'Normalized powers of HF', 'LF/HF ratio',
         'Sample entropy', 'Lyapunov exponent', 'Valence', 'Arousal', 'Emotion']
ecgFeatures_T = ecgFeatures.T
num_plot = 9
if ecgFeatures_T.shape[0] % num_plot == 0:
    num_figure = ecgFeatures_T.shape[0] // num_plot
else:
    num_figure = ecgFeatures_T.shape[0] // num_plot + 1

for i in range(num_figure):
    plt.figure(figsize=(12, 9))
    for j in range(num_plot):
        if num_plot*i+j >= ecgFeatures_T.shape[0]:
            break
        plt.subplot(3, 3, j + 1)
        plt.plot(ecgFeatures_T[num_plot*i+j])
        # plt.xlabel('Time [s]')
        plt.title(title[num_plot*i+j])
    plt.tight_layout()
plt.show()

