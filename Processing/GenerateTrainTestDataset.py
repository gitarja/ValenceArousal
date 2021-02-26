import pandas as pd
import glob
# from Conf.Settings import STRIDE

data_path = "G:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\"
dataframe = pd.DataFrame(columns=["Idx", "Start", "End", "Valence", "Arousal", "Emotion", "Status", "Subject"])
STRIDE = 1.0
TEST_SPLIT = 0.2

for folder in glob.glob(data_path + "*"):
    for subject in glob.glob(folder + "\\*-2020-*"):
        try:
            data = pd.read_csv(subject + "\\features_list_" + str(STRIDE) + ".csv")
            dataframe = pd.concat([dataframe, data], axis=0, ignore_index=True)
        except:
            print("Error:", subject)

dataframe = dataframe.sample(frac=1).reset_index(drop=True)
train_dataset = dataframe[:-int(len(dataframe) * TEST_SPLIT)]
test_dataset = dataframe[-int(len(dataframe) * TEST_SPLIT):]

train_dataset.to_csv(data_path + "training_" + str(STRIDE) + ".csv", index=False)
test_dataset.to_csv(data_path + "testing_" + str(STRIDE) + ".csv", index=False)
