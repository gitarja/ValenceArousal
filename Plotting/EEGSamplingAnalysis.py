import pandas as pd
from matplotlib import pyplot as plt

path = "D:\\usr\\pras\\data\YAMAHA\\Yamaha 国大\\TS103_20200617_141440_279\\"
data = pd.read_csv(path+"20200617_141440_279_HB_PW.csv")
n=7500
for i in range(int(data.__len__()/n)):
    ecg_data = data.iloc[i*n:(i+1)*n]["ecg"].values

    plt.plot(ecg_data)
    plt.savefig(path+str(i)+"_ecg.png")
    plt.close()
    # print(ecg_data)