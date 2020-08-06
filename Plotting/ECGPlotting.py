import pandas as pd
from Libs.Utils import utcToTimeStamp, timeToInt, butterBandpassFilter, windowFilter
from matplotlib import pyplot as plt
from ECG.ECGFeatures import ECGFeatures

path ="D:\\usr\\pras\\data\\EmotionTestVR\\Watanabe\\"
save_path ="D:\\usr\\pras\\result\\ValenceArousal\\ECG\\"
game_results_file = path + "Watanabe_M_2020_7_27_11_25_52_gameResults.csv"
ecg_file = path + "20200727_112502_148_HB_PW.csv"

ecg_data = pd.read_csv(ecg_file)
ecg_data["timestamp"] = ecg_data["timestamp"].apply(timeToInt)

game_results = pd.read_csv(game_results_file)
featuresExct = ECGFeatures()
for i in (0, 1, 7, 9):
    time_start = timeToInt(game_results.iloc[i]["Time_Start"])
    time_end = timeToInt(game_results.iloc[i]["Time_End"])

    ecg = ecg_data[(ecg_data["timestamp"].values>=time_start) & (ecg_data["timestamp"].values <=time_end)]["ecg"].values


    ts, hb = featuresExct.computeHeartBeat(ecg, 256.)

    plt.plot(ts, hb)
    plt.ylim([50, 120])
    plt.ylabel("Heart beat (bmp)")
    plt.savefig(save_path + str(i)+"_ecg.png")
    plt.show()

