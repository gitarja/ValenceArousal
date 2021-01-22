import pandas as pd


def convertLabel(ar, val, th=0.5):
    if (ar > th) and (val > th):
        return 3
    if (ar > th) and (val < th):
        return 0
    if (ar < th) and (val < th):
        return 1
    if (ar < th) and (val > th):
        return 2

path = "D:\\usr\\pras\\data\\driver\\E6\\"
data = pd.read_csv(path + "E6_NOCITY.csv")

labels = []
for index, row in data.iterrows():
    label = convertLabel(row["arousal"], row["valence"])
    labels.append(label)


results = pd.DataFrame({"latitude":data["latitude"].values,	"longitude":data["longitude"].values, "color": labels})

results.to_csv(path +"E6_NOCITY_labels.csv", index=False)
