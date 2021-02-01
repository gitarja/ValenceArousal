from EEG.EEGFeatures import EEGFeatures
import pandas as pd
import glob
from Libs.Utils import timeToInt, arToLabels
from Conf.Settings import FS_EEG, SPLIT_TIME, STRIDE, EXTENTION_TIME, EEG_RAW_PATH, EEG_PATH, DATASET_PATH
from EEG.SpaceLapFilter import SpaceLapFilter
import numpy as np
from scipy import signal

data_path = "D:\\usr\\pras\\data\\YAMAHA\\Yamaha-Experiment (2020-10-26 - 2020-11-06)\\data\\*"
eeg_file = "\\EEG\\"
game_result = "\\*_gameResults.csv"
min_eeg_len = SPLIT_TIME * FS_EEG - 100
downsample_eeg_len = SPLIT_TIME * 200
for folder in glob.glob(DATASET_PATH + "*"):

    for subject in glob.glob(folder + "\\*-2020-*"):
        print(subject)
        try:
            data_EmotionTest = pd.read_csv(glob.glob(subject + game_result)[0])

            data_EmotionTest.loc[:, 'Time_Start'] = data_EmotionTest.loc[:, 'Time_Start'].apply(timeToInt)
            data_EmotionTest.loc[:, 'Time_End'] = data_EmotionTest.loc[:, 'Time_End'].apply(timeToInt)

            eeg_features_list = pd.DataFrame(
                columns=["Idx", "Start", "End", "Valence", "Arousal", "Emotion", "Status", "Subject"])
            idx = 0
            eeg_filter = SpaceLapFilter()
            eeg_features_exct = EEGFeatures(FS_EEG)
            for i in range(len(data_EmotionTest)):

                tdelta = data_EmotionTest.iloc[i]["Time_End"] - data_EmotionTest.iloc[i]["Time_Start"]
                time_end = (data_EmotionTest.iloc[i]["Time_End"])
                valence = data_EmotionTest.iloc[i]["Valence"]
                arousal = data_EmotionTest.iloc[i]["Arousal"]
                emotion = data_EmotionTest.iloc[i]["Emotion"]
                eeg = pd.read_csv(subject + eeg_file + "filtered_eeg" + str(i) + ".csv")
                print(i)
                eeg["time"] = eeg["time"].apply(timeToInt)
                # print(eeg["time"].values[0])


                #setting the end of extraction
                bin_ar = arToLabels(arousal)
                bin_val = arToLabels(valence)

                if (bin_ar == 1) or (bin_val == 1):
                    end_extract = 0.5 * (tdelta // SPLIT_TIME)#use only half of the data from the mid to  the last
                else:
                    end_extract = 0.3 * (tdelta // SPLIT_TIME)#use only 2/3 of the data

                for j in np.arange(end_extract, (tdelta // SPLIT_TIME), STRIDE): #the window move backward
                    # take 2.5 sec after end
                    # end = time_end - ((j-1) * SPLIT_TIME) + EXTENTION_TIME
                    # start = time_end - (j * SPLIT_TIME)


                    end = time_end - ((j) * SPLIT_TIME)
                    start = time_end - ((j+1) * SPLIT_TIME) - EXTENTION_TIME

                    eeg_split = eeg[(eeg["time"].values >= start) & (
                            eeg["time"].values <= end)]
                    eeg_filtered = eeg_split.loc[:, "CH0":"CH18"].values[:min_eeg_len]
                    status = 0
                    # print("Start:" + str(start) + " End:"+ str(end) +  " Shape:"+str(eeg_filtered.shape[0]))
                    if eeg_filtered.shape[0] >= min_eeg_len:
                        time_domain_features = eeg_features_exct.extractTimeDomainAll(eeg_filtered)
                        freq_domain_features = eeg_features_exct.extractFrequencyDomainAll(eeg_filtered)
                        plf_features = eeg_features_exct.extractPLFFeatures(eeg_filtered)
                        power_features = eeg_features_exct.extractPowerFeatures(eeg_filtered)
                        if (time_domain_features.shape[0] != 0) and (freq_domain_features.shape[0] != 0) :
                            eeg_features = np.concatenate([time_domain_features, freq_domain_features, plf_features, power_features])
                            if np.sum(np.isinf(eeg_features)) == 0 and np.sum(np.isinf(eeg_features)) == 0:
                                np.save(subject + EEG_PATH + "eeg_" + str(idx) + ".npy", eeg_features)
                                # np.save(subject + EEG_RAW_PATH + "eeg_" + str(idx) + ".npy", signal.resample(eeg_filtered, downsample_eeg_len))
                                status = 1

                        # add object to dataframes
                    subject_name = subject.split("\\")[-1]
                    eeg_features_list = eeg_features_list.append(
                            {"Idx": str(idx), "Subject": subject_name, "Start": str(start), "End": str(end),
                             "Valence": valence, "Arousal": arousal,
                             "Emotion": emotion, "Status": status},
                            ignore_index=True)
                    idx += 1

            eeg_features_list.to_csv(subject + "\\EEG_features_list_"+str(STRIDE)+".csv", index=False)

        except:
            print("Error: " + subject)
