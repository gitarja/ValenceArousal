import pandas as pd
from Libs.Utils import utcToTimeStamp, timeToInt, butterBandpassFilter
from matplotlib import pyplot as plt
from GSR.GSRFeatures import GSRFeatures

path ="D:\\usr\\pras\\data\\EmotionTestVR\\Watanabe\\"
save_path ="D:\\usr\\pras\\result\\ValenceArousal\\GSR+PPG\\"
game_results_file = path + "Watanabe_M_2020_7_27_11_25_52_gameResults.csv"
gsr_file = path + "Watanabe_GSR(correct).csv"

gsr_data = pd.read_csv(gsr_file, header=[0,1])
gsr_data["GSR_Timestamp_Shimmer_CAL"] = gsr_data["GSR_Timestamp_Shimmer_CAL"].apply(utcToTimeStamp, axis=1)

game_results = pd.read_csv(game_results_file)
featuresExct = GSRFeatures()
for i in (0, 1, 7, 9):
    time_start = timeToInt(game_results.iloc[i]["Time_Start"])
    time_end = timeToInt(game_results.iloc[i]["Time_End"])

    skin_conductance = gsr_data[(gsr_data["GSR_Timestamp_Shimmer_CAL"].values>=time_start) & (gsr_data["GSR_Timestamp_Shimmer_CAL"].values <=time_end)]["GSR_GSR_Skin_Conductance_CAL"].values
    ppg = gsr_data[(gsr_data["GSR_Timestamp_Shimmer_CAL"].values>=time_start) & (gsr_data["GSR_Timestamp_Shimmer_CAL"].values <=time_end)]["GSR_PPG_A13_CAL"].values

    skin_conductance_f = butterBandpassFilter(skin_conductance, 0.03, 5.0, 15.0, order=5)
    ts, ppg_f = featuresExct.computeHeartBeat(ppg.flatten(), 15.0)

    plt.plot(skin_conductance_f)
    plt.ylim([0, 1])
    plt.ylabel("Skin conductance (µS)")
    plt.savefig(save_path+str(i)+"_GSR.png")
    plt.show()
    plt.plot(ts, ppg_f)
    plt.ylim([40, 100])
    plt.ylabel("Heart rate (bpm)")
    plt.savefig(save_path+str(i)+"_PPG.png")
    plt.show()

