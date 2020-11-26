import glob
from Conf.Settings import FS_ECG, FS_GSR, FS_RESP
from ECG.ECGFeatures import ECGFeatures
from GSR.GSRFeatures import PPGFeatures, EDAFeatures
from Resp.RespFeatures import RespFeatures
import pandas as pd

data_path = "D:\\usr\\pras\\data\\YAMAHA\\Yamaha-Experiment (2020-10-26 - 2020-11-06)\\data\\*"
ecg_file = "\\ECG\\"
gsr_file = "\\GSR\\"
resp_file = "\\Resp\\"

ecg_features = ECGFeatures(fs=FS_ECG)
ppg_features = PPGFeatures(fs=FS_GSR)
eda_features = EDAFeatures(fs=FS_GSR)
resp_features = RespFeatures(fs=FS_RESP)


for folder in glob.glob(data_path):
    for subject in glob.glob(folder + "\\*-2020-*"):
        print(subject)
        try:
            # filter ecg iPhone
            # ecg = pd.read_csv(glob.glob(subject+ecg_file+"*.csv")[0])
            # ecg_filtered = ecg_features.filterECG(ecg["ecg"].values)
            # ecg["ecg"] = ecg_filtered
            # ecg.to_csv(subject+ecg_file+"filtered_ecg.csv", index=False)
            #
            # # filter PPG and EDA
            gsr = pd.read_csv(glob.glob(subject + gsr_file + "*.csv")[0],  header=[0,1])
            time = gsr.iloc[:, 0].values
            eda = gsr.iloc[:, 5].values
            ppg = gsr.iloc[:, 7].values
            filtered_eda = eda_features.filterEDA(eda)
            filtered_ppg = ppg_features.filterPPG(ppg)

            eda_final = pd.DataFrame({"time": time, "eda": filtered_eda})
            ppg_final = pd.DataFrame({"time": time, "ppg": filtered_ppg})

            eda_final.to_csv(subject + gsr_file +"filtered_eda.csv", index=False)
            ppg_final.to_csv(subject + gsr_file +"filtered_ppg.csv", index=False)

            # filter ECG and Resp
            # ecg_resp = pd.read_csv(glob.glob(subject + resp_file + "*.csv")[0], header=[0, 1])
            #
            # time = ecg_resp.iloc[:, 0].values
            # ecg = ecg_resp.iloc[:, 8].values
            # resp = ecg_resp.iloc[:, 9].values
            #
            # filtered_ecg_resp = ecg_features.waveDriftFilter(ecg_features.filterECG(ecg), n=9)
            # filtered_resp = resp_features.filterResp(resp)
            #
            # ecg_final = pd.DataFrame({"time": time, "ecg": filtered_ecg_resp})
            # resp_final = pd.DataFrame({"time": time, "resp": filtered_resp})
            #
            # ecg_final.to_csv(subject + resp_file + "filtered_ecg_resp.csv", index=False)
            # resp_final.to_csv(subject + resp_file + "filtered_resp.csv", index=False)
        except:
            print("Error: "+ subject)
