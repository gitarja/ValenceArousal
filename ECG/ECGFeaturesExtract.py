from ECG.ECGFeatures import ECGFeatures
import pandas as pd
from Libs.Utils import timeToInt
from scipy import stats
import numpy as np
from Conf import Settings as set
import matplotlib
import matplotlib.pyplot as plt


subject = "Komiyama"
path = "C:\\Users\\ShimaLab\\Documents\\nishihara\\data\\EmotionTestVR\\" + subject + "\\"
path_results = path + 'results\\ECG\\'
path_raw = path + 'results\\ECG_raw\\'
path_ECGdata = path + 'Komiya_ECG.csv'
path_EmotionTest = path + 'Komiya_M_2020_7_9_15_22_44_gameResults.csv'


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

# features extractor
featuresExct = ECGFeatures(set.FS_ECG)
emotionTestResult = pd.DataFrame(columns=["Idx", "Start", "End", "Valence", "Arousal", "Emotion", "Status", "Subject"])
split_time = 45
idx = 0
for i in range(len(data_EmotionTest)):
    print('{}/{}'.format(i + 1, len(data_EmotionTest)))
    tdelta = data_EmotionTest.iloc[i]["Time_End"] - data_EmotionTest.iloc[i]["Time_Start"]
    time_end = data_EmotionTest.iloc[i]["Time_End"]
    valence = data_EmotionTest.iloc[i]["Valence"]
    arousal = data_EmotionTest.iloc[i]["Arousal"]
    emotion = data_EmotionTest.iloc[i]["Emotion"]

    for j in np.arange(0, (tdelta // split_time), 0.4):
        end = time_end - (j * split_time)
        start = time_end - ((j + 1) * split_time)
        ecg = data[(data["timestamp"].values >= start) & (data["timestamp"].values <= end)]
        status = 0

        #save raw ecg data
        np.save(path_raw + "ecg_raw_" + str(idx) + ".npy", ecg['ecg'].values)
        # extract ecg features
        time_domain = featuresExct.extractTimeDomain(ecg['ecg'].values)
        freq_domain = featuresExct.extractFrequencyDomain(ecg['ecg'].values)
        nonlinear_domain = featuresExct.extractNonLinearDomain(ecg['ecg'].values)
        if time_domain.shape[0] != 0 and freq_domain.shape[0] != 0 and nonlinear_domain.shape[0] != 0:
            featuresEachMin = np.concatenate([time_domain, freq_domain, nonlinear_domain])
            if np.sum(np.isinf(featuresEachMin)) == 0 | np.sum(np.isnan(featuresEachMin)) == 0:
                np.save(path_results + "ecg_" + str(idx) + ".npy", featuresEachMin)
                status = 1
            else:
                status = 0

        # add object to dataframes
        emotionTestResult = emotionTestResult.append(
            {"Idx": str(idx), "Subject": subject, "Start": str(start), "End": str(end), "Valence": valence, "Arousal": arousal, "Emotion": emotion, "Status": status},
            ignore_index=True)
        idx += 1

# save to csv
emotionTestResult.to_csv(path + 'ECG_features_list.csv')

'''
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
'''
