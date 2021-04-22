import pandas as pd
import glob

def convertLabel(ar, val, th=0.5):
    if (ar >= th) and (val >= th):
        return 3
    if (ar >= th) and (val < th):
        return 0
    if (ar < th) and (val < th):
        return 1
    if (ar < th) and (val >= th):
        return 2

def convertLabelThree(ar, val, th=0.5):
    if abs(val) <= th:
        return 0
    elif val > th:
        if ar > th:
            return 3
        elif abs(ar) <= th:
            return 2
        elif ar < -th:
            return 1
    elif val < -th:
        if ar > th:
            return 6
        elif abs(ar) <= th:
            return 5
        elif ar < -th:
            return 4


path = "D:\\usr\\pras\\data\\driver\\2021-04-20\\*"
for folder in glob.glob(path):
    for file in glob.glob(folder + "\\*_results.csv"):
        print(file)
        data = pd.read_csv(file)
        file_name = file.split("\\")[-1]
        result_file_name = file_name.replace("_results.csv", "_results_labels.csv")
        labels = []
        for index, row in data.iterrows():
            label = convertLabelThree(row["arousal"], row["valence"])
            labels.append(label)


        results = pd.DataFrame({"latitude":data["latitude"].values,	"longitude":data["longitude"].values, "color": labels})

        results.to_csv(folder + "\\" +result_file_name, index=False)
