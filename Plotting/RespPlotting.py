import pandas as pd
from Libs.Utils import utcToTimeStamp, timeToInt, butterBandpassFilter, windowFilter
from matplotlib import pyplot as plt
import biosppy
path ="D:\\usr\\pras\\data\\EmotionTestVR\\Watanabe\\"
save_path ="D:\\usr\\pras\\result\\ValenceArousal\\Resp+ECG\\"
game_results_file = path + "Watanabe_M_2020_7_27_11_25_52_gameResults.csv"
gsr_file = path + "Watanabe_resp.csv"

resp_data = pd.read_csv(gsr_file, header=[0,1])
resp_data["RESP_Timestamp_Unix_CAL"] = resp_data["RESP_Timestamp_Unix_CAL"].apply(utcToTimeStamp, axis=1)

game_results = pd.read_csv(game_results_file)
for i in (0, 1, 7, 9):
    time_start = timeToInt(game_results.iloc[i]["Time_Start"])
    time_end = timeToInt(game_results.iloc[i]["Time_End"])

    resp = resp_data[(resp_data["RESP_Timestamp_Unix_CAL"].values>=time_start) & (resp_data["RESP_Timestamp_Unix_CAL"].values <=time_end)]["RESP_ECG_RESP_24BIT_CAL"].values
    ll_ra = resp_data[(resp_data["RESP_Timestamp_Unix_CAL"].values>=time_start) & (resp_data["RESP_Timestamp_Unix_CAL"].values <=time_end)]["RESP_ECG_LL-RA_24BIT_CAL"].values

    resp_f = windowFilter(resp, 120, 2.0, fs=256.)
    ll_ra_f = windowFilter(ll_ra, 120, 30., fs=256.)
    ts, resp_rate = biosppy.signals.resp.resp(resp.flatten(), sampling_rate=256., show=False)[3:]

    plt.plot(resp_f)
    plt.savefig(save_path + str(i)+"_resp.png")
    plt.show()
    plt.plot(ll_ra_f)
    plt.savefig(save_path + str(i)+"_ecg_ll-ra.png")
    plt.show()
    plt.plot(ts, resp_rate)
    plt.ylim([0, 0.35])
    plt.ylabel("Respiration rate Hz")
    plt.savefig(save_path + str(i) + "_resp_rate.png")
    plt.show()

