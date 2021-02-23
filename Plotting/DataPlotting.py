import glob
import os
from Conf.Settings import FS_ECG, FS_GSR, FS_RESP, SPLIT_TIME
from ECG.ECGFeatures import ECGFeatures
from GSR.GSRFeatures import PPGFeatures, EDAFeatures
from Resp.RespFeatures import RespFeatures
from Libs.Utils import timeToInt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_path = "G:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\*"
ecg_file = "\\ECG\\filtered_ecg.csv"
eda_file = "\\GSR\\filtered_eda.csv"
ppg_file = "\\GSR\\filtered_ppg.csv"
resp_file = "\\Resp\\filtered_resp.csv"
ecg_resp_file = "\\Resp\\filtered_ecg_resp.csv"
game_result = "\\*_gameResults.csv"
plot_result = "\\plot_eps\\"

ecg_features = ECGFeatures(fs=FS_ECG)
ppg_features = PPGFeatures(fs=FS_GSR)
eda_features = EDAFeatures(fs=FS_GSR)
resp_features = RespFeatures(fs=FS_RESP)
min_len = FS_RESP * (SPLIT_TIME + 1)

for folder in glob.glob(data_path):
    for subject in glob.glob(folder + "\\*A6-2020-*"):
        # try:
        data_EmotionTest = pd.read_csv(glob.glob(subject + game_result)[0])
        data_EmotionTest["Time_Start"] = data_EmotionTest["Time_Start"].apply(timeToInt)
        data_EmotionTest["Time_End"] = data_EmotionTest["Time_End"].apply(timeToInt)
        # ecg
        ecg = pd.read_csv(subject + ecg_file)
        ecg.iloc[:, 0] = ecg.iloc[:, 0].apply(timeToInt)
        # ppg
        ppg = pd.read_csv(subject + ppg_file)
        ppg.iloc[:, 0] = ppg.iloc[:, 0].apply(timeToInt)
        # eda
        eda = pd.read_csv(subject + eda_file)
        eda.iloc[:, 0] = eda.iloc[:, 0].apply(timeToInt)

        # resp
        resp = pd.read_csv(subject + resp_file)
        resp.iloc[:, 0] = resp.iloc[:, 0].apply(timeToInt)

        #ecg _resp
        ecg_resp = pd.read_csv(subject + ecg_resp_file)
        ecg_resp.iloc[:, 0] = ecg_resp.iloc[:, 0].apply(timeToInt)

        for i in range(len(data_EmotionTest)):
            tdelta = data_EmotionTest.iloc[i]["Time_End"] - data_EmotionTest.iloc[i]["Time_Start"]
            time_end = data_EmotionTest.iloc[i]["Time_End"]
            valence = data_EmotionTest.iloc[i]["Valence"]
            arousal = data_EmotionTest.iloc[i]["Arousal"]
            idx = 0
            for j in np.arange(0, (tdelta // SPLIT_TIME), 0.4):
                end = time_end - (j * SPLIT_TIME)
                start = time_end - ((j + 1) * SPLIT_TIME)
                # split and plot
                ecg_split = ecg[(ecg.iloc[:, 0].values >= start) & (
                        ecg.iloc[:, 0].values <= end)].values[:, 1]
                ppg_split = ppg[(ppg.iloc[:, 0].values >= start) & (
                        ppg.iloc[:, 0].values <= end)].values[:, 1]
                eda_split = eda[(eda.iloc[:, 0].values >= start) & (
                        eda.iloc[:, 0].values <= end)].values[:, 1]
                resp_split = resp[(resp.iloc[:, 0].values >= start) & (
                        resp.iloc[:, 0].values <= end)].values[:, 1]
                ecg_resp_split = ecg_resp[(ecg_resp.iloc[:, 0].values >= start) & (
                        ecg_resp.iloc[:, 0].values <= end)].values[:, 1]

                fig = plt.figure()
                plt.subplot(411)
                plt.plot(np.linspace(0, SPLIT_TIME, len(ecg_split)), ecg_split)
                plt.ylabel("Amplitude")
                plt.tick_params(labelbottom=False)
                # plt.subplot(512)
                # plt.plot(ecg_resp_split)
                plt.subplot(412)
                plt.plot(np.linspace(0, SPLIT_TIME, len(ppg_split)), ppg_split)
                plt.ylabel("Amplitude")
                plt.tick_params(labelbottom=False)
                plt.subplot(413)
                plt.plot(np.linspace(0, SPLIT_TIME, len(eda_split)), eda_split)
                plt.ylabel("Amplitude")
                plt.tick_params(labelbottom=False)
                plt.subplot(414)
                plt.plot(np.linspace(0, SPLIT_TIME, len(resp_split)), resp_split)
                plt.ylabel("Amplitude")
                plt.xlabel("Time [s]")

                fig.tight_layout()
                fig.align_labels()
                # plt.show()

                os.makedirs(subject + plot_result, exist_ok=True)
                fig.savefig(subject + plot_result + str(i) + "_" + str(idx) + "_" + str(valence) +"_"+str(arousal)+".eps")
                plt.close()
                idx+=1

    # except:
    #     print(subject)
