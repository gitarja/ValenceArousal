import pandas as pd
from Conf.Settings import DATASET_PATH
import os


data_path = DATASET_PATH + "stride=0.2\\"
all_data = pd.read_csv(data_path + "all_data.csv")
train_data = pd.DataFrame(columns=all_data.columns)
val_data = pd.DataFrame(columns=all_data.columns)
test_data = pd.DataFrame(columns=all_data.columns)
val_subjects = ["C4-2020-10-29", "D3-2020-10-30"]
test_subjects = ["C2-2020-11-04", "D4-2020-11-06"]

for i in range(len(all_data)):
    if all_data.iloc[i]["Subject"] in val_subjects:
        val_data = val_data.append(all_data.iloc[i], ignore_index=True)
    elif all_data.iloc[i]["Subject"] in test_subjects:
        test_data = test_data.append(all_data.iloc[i], ignore_index=True)
    else:
        train_data = train_data.append(all_data.iloc[i], ignore_index=True)

result_path = data_path + "SubjectCV\\"
os.makedirs(result_path, exist_ok=True)
train_data.to_csv(result_path + "training_data_1.csv", index=False)
val_data.to_csv(result_path + "validation_data_1.csv", index=False)
test_data.to_csv(result_path + "test_data_1.csv", index=False)
