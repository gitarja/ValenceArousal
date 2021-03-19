import pandas as pd
import numpy as np
from Conf.Settings import DREAMER_PATH
import glob

label_all = pd.DataFrame(columns=["Subject", "VideoSequence", "Arousal", "Valence"])
for sub in range(1, len(glob.glob(DREAMER_PATH + "subject_*\\")) + 1):
    path = DREAMER_PATH + "subject_" + str(sub) + "\\"
    arousal = pd.read_csv(path + "score_arousal_" + str(sub) + ".csv", names=["Arousal"])
    valence = pd.read_csv(path + "score_valence_" + str(sub) + ".csv", names=["Valence"])
    video_seq = pd.DataFrame(np.arange(1, len(arousal) + 1), columns=["VideoSequence"])
    subject = pd.DataFrame(np.tile(sub, len(arousal)), columns=["Subject"])
    label = pd.concat([subject, video_seq, arousal, valence], axis=1)
    label_all = pd.concat([label_all, label])

label_all.to_csv(DREAMER_PATH + "labels.csv", index=False)
