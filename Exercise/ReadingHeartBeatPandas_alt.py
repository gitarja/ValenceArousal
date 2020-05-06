
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

#timestampt to int
def timeToInt(time):
    date, hours = time.split(" ")
    h, m, s = hours.split(":")
    s, ms = s.split(".")
    inttime = 3600 * float(h) + 60 * float(m) + float(s) + 0.000001 * float(ms)

    return inttime



# ファイルから心拍データを読み込む(Pandas使用)
filename = "../Data/Dummy/hb_data.csv"
data = pd.read_csv(filename)

#convert timestamp to int
data["timestamp"] = data["timestamp"].apply(timeToInt)

#normalize the data
data["timestamp"] = data["timestamp"]- data.loc[0].timestamp

#group for every 2 minutes
#250HZ = ECG sensor rate
groups = data.groupby(data["timestamp"].values // 120.)
i = 0
for g in groups:
    data = g[1]
    plt.figure(i)
    plt.plot( data.index/250, data.ecg)
    plt.title("Heartbeat")
    plt.xlabel("Time t[s]")
    i+=1

plt.show()