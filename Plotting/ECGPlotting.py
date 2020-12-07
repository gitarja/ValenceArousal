import glob
from ECG.ECGFeatures import ECGFeatures
from GSR.GSRFeatures import PPGFeatures, EDAFeatures
from Resp.RespFeatures import RespFeatures
from Libs.Utils import timeToInt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_path = "D:\\usr\\pras\\data\\YAMAHA\\Yamaha-Road-Test\\20201119ロードテストE組データ\\*"
SPLIT_TIME = 60
for subject in glob.glob(data_path):
    for file in glob.glob(subject + "\\*_HB_PW.csv"):
        filename = file.split("\\")[-1].split("_HB_PW.csv")[0]
        ecg_file = pd.read_csv(file)
        ecg_file["timestamp"] = ecg_file["timestamp"].apply(timeToInt)
        tstart = ecg_file.iloc[0]["timestamp"]
        tdelta = ecg_file.iloc[-1]["timestamp"] - ecg_file.iloc[0]["timestamp"]
        for j in np.arange(0, (tdelta // SPLIT_TIME), 1.):
            start = (j * SPLIT_TIME) + tstart
            end = (j+1) * SPLIT_TIME + tstart
            ecg_split = ecg_file[(ecg_file["timestamp"] >= start) & (ecg_file["timestamp"] <= end)]["ecg"].values
            print(len(ecg_split))
            plt.figure()
            plt.plot(ecg_split)
            plt.savefig(subject + "\\" + filename +"_"+ str(int(j)) + ".png")
            plt.close()