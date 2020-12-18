import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

EDA = 1102
PPG = 11
Resp = 11
ECG_Resp = 13
ECG = 13
EEG = 1330

FEATURES_N = 2480

EDA_LABELS = ["EDA" for x in range(EDA)]
PPG_LABELS = ["PPG" for x in range(PPG)]
Resp_LABELS = ["Resp" for x in range(Resp)]
ECG_Resp_LABELS = ["ECG_Resp" for x in range(ECG_Resp)]
ECG_LABELS = ["ECG" for x in range(ECG)]
EEG_LABELS = ["EEG" for x in range(EEG)]

AROUSAL_LABELS = ["Arousal" for x in range(FEATURES_N)]
VALENCE_LABELS = ["Valence" for x in range(FEATURES_N)]

labels = np.concatenate([EDA_LABELS, PPG_LABELS, Resp_LABELS, ECG_Resp_LABELS, ECG_LABELS, EEG_LABELS])

labels = np.concatenate([labels, labels])
ar_val_labels = np.concatenate([AROUSAL_LABELS, VALENCE_LABELS])

#Prepare Anova Results
#F
anova_ar = np.loadtxt("..\\StatisticalAnalysis\\results\\f_ar.csv")
anova_val = np.loadtxt("..\\StatisticalAnalysis\\results\\f_val.csv")

anova_coeff = np.concatenate([anova_ar, anova_val])

anova_d = {"Anova-F": anova_coeff, "Sensor": labels, "AR_VAL":ar_val_labels}
anova_data = pd.DataFrame(anova_d)

#P
anova_ar_p = np.loadtxt("..\\StatisticalAnalysis\\results\\p_ar.csv")
anova_val_p = np.loadtxt("..\\StatisticalAnalysis\\results\\p_val.csv")

anova_p_coeff = np.concatenate([anova_ar_p, anova_val_p])

anova_p_d = {"Anova-P": anova_p_coeff, "Sensor": labels, "AR_VAL":ar_val_labels}
anova_p_data = pd.DataFrame(anova_p_d)

#Prepare MI Results
mi_ar = np.loadtxt("..\\StatisticalAnalysis\\results\\mi_ar.csv")
mi_val = np.loadtxt("..\\StatisticalAnalysis\\results\\mi_val.csv")

mi_coeff = np.concatenate([mi_ar, mi_val])

mi_d = {"MI": mi_coeff, "Sensor": labels, "AR_VAL":ar_val_labels}
mi_data = pd.DataFrame(mi_d)

#Prepare MI Results
rf_ar = np.loadtxt("..\\StatisticalAnalysis\\results\\rf_ar.csv")
rf_val = np.loadtxt("..\\StatisticalAnalysis\\results\\rf_val.csv")

rf_coeff = np.concatenate([rf_ar, rf_val])

rf_d = {"FI": rf_coeff, "Sensor": labels, "AR_VAL":ar_val_labels}
rf_data = pd.DataFrame(rf_d)


#Plotting
#Anova-F
plt.figure(1)
sns.boxplot(x="Sensor", y="Anova-F",  hue="AR_VAL",
                 data=anova_data, palette="Set2").set_title('Anova-F')

#Anova-P
plt.figure(2)
sns.boxplot(x="Sensor", y="Anova-P",  hue="AR_VAL",
                 data=anova_p_data, palette="Set2").set_title('Anova-P')
#MI
plt.figure(3)
sns.boxplot(x="Sensor", y="MI",  hue="AR_VAL",
                 data=mi_data, palette="Set2").set_title('MI')

#RF
plt.figure(4)
sns.boxplot(x="Sensor", y="FI",  hue="AR_VAL",
                 data=rf_data, palette="Set2").set_title('RF')


plt.show()
