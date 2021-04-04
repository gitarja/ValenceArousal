import glob
from Conf.Settings import FS_ECG, FS_GSR, FS_RESP, SPLIT_TIME
from ECG.ECGFeatures import ECGFeatures
from GSR.GSRFeatures import PPGFeatures, EDAFeatures
from Resp.RespFeatures import RespFeatures
from Libs.Utils import timeToInt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

data_path = "D:\\usr\\pras\\data\\YAMAHA\\Yamaha-Experiment (2020-10-26 - 2020-11-06)\\data\\*"
eeg_file = "\\EEG\\"
game_result = "\\*_gameResults.csv"
plot_result = "\\plot_eeg\\"

ecg_features = ECGFeatures(fs=FS_ECG)
ppg_features = PPGFeatures(fs=FS_GSR)
eda_features = EDAFeatures(fs=FS_GSR)
resp_features = RespFeatures(fs=FS_RESP)
min_len = FS_RESP * (SPLIT_TIME + 1)

for folder in glob.glob(data_path):
    for subject in glob.glob(folder + "\\*-2020-*"):

        data_EmotionTest = pd.read_csv(glob.glob(subject + game_result)[0])
        data_EmotionTest["Time_Start"] = data_EmotionTest["Time_Start"].apply(timeToInt)
        data_EmotionTest["Time_End"] = data_EmotionTest["Time_End"].apply(timeToInt)
        print(subject)
        for i in range(len(data_EmotionTest)):
            eeg = pd.read_csv(subject + eeg_file + "filtered_eeg" + str(i) + ".csv")
            eeg.iloc[:, 0] = eeg.iloc[:, 0].apply(timeToInt)

            tdelta = data_EmotionTest.iloc[i]["Time_End"] - data_EmotionTest.iloc[i]["Time_Start"]
            time_end = data_EmotionTest.iloc[i]["Time_End"]
            valence = data_EmotionTest.iloc[i]["Valence"]
            arousal = data_EmotionTest.iloc[i]["Arousal"]
            idx = 0
            for j in np.arange(0, (tdelta // SPLIT_TIME), 0.4):

                try:
                    end = time_end - (j * SPLIT_TIME)
                    start = time_end - ((j + 1) * SPLIT_TIME)
                    # split and plot
                    eeg_split = eeg[(eeg.iloc[:, 0].values >= start) & (
                            eeg.iloc[:, 0].values <= end)].values[:, 1:]

                    # plot data

                    # Plot the EEG
                    fig = plt.figure("MRI_with_EEG")
                    n_samples, n_rows = eeg_split.shape
                    t = 10 * np.arange(n_samples) / n_samples
                    ticklocs = []
                    ax2 = fig.add_subplot(1, 1, 1)
                    ax2.set_xlim(0, 10)
                    ax2.set_xticks(np.arange(10))
                    dmin = eeg_split.min()
                    dmax = eeg_split.max()
                    dr = (dmax - dmin) * 0.7  # Crowd them a bit.
                    y0 = dmin
                    y1 = (n_rows - 1) * dr + dmax
                    ax2.set_ylim(y0, y1)

                    segs = []
                    y_labels = []
                    for k in range(n_rows):
                        segs.append(np.column_stack((t, eeg_split[:, k])))
                        ticklocs.append(k * dr)
                        y_labels.append("CH" + str(k))

                    offsets = np.zeros((n_rows, 2), dtype=float)
                    offsets[:, 1] = ticklocs

                    lines = LineCollection(segs, offsets=offsets, transOffset=None)
                    ax2.add_collection(lines)

                    ax2.set_yticks(ticklocs)
                    ax2.set_yticklabels(y_labels)

                    plt.savefig(
                        subject + plot_result + str(i) + "_" + str(idx) + "_" + str(valence) + "_" + str(
                            arousal) + ".png")
                    plt.close()
                    idx += 1
                except:
                    print("Error: " +subject)
