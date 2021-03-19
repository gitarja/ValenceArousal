import glob
from Conf.Settings import DREAMER_PATH, NUM_SUBJECTS, NUM_VIDEO_SEQ
from ECG.ECGFeatures import ECGFeatures
import pandas as pd

FS_ECG = 256
ecg_file = "ECG\\"

ecg_features = ECGFeatures(fs=FS_ECG)

# for s, subject in enumerate(glob.glob(DREAMER_PATH + "subject_*\\")):
for subject in range(1, NUM_SUBJECTS + 1):
    print("subject_" + str(subject))
    subject_path = DREAMER_PATH + "subject_" + str(subject) + "\\"

    # filter ecg
    # for i, file in enumerate(glob.glob(subject + ecg_file + "*.csv")):
    for video in range(1, NUM_VIDEO_SEQ + 1):
        ecg = pd.read_csv(subject_path + ecg_file + "ecg_" + str(subject) + "_" + str(video) + ".csv",
                          names=("CH1", "CH2"))
        for ch in ecg.columns:
            ecg_filtered = ecg_features.filterECG(ecg[ch].values)
            ecg[ch] = ecg_filtered

        ecg.to_csv(subject_path + ecg_file + "filtered_ecg_" + str(subject) + "_" + str(video) + ".csv", index=False)

