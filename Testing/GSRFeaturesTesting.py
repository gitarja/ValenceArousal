from GSR.GSRFeatures import PPGFeatures, EDAFeatures
from Conf import Settings as set
import pandas as pd

data = pd.read_csv("..\\Data\\Dummy\\GSR_sample.csv", header=[0,1])

#PPG
ppg = data["GSR_PPG_A13_CAL"].iloc[0:900].values

ppgExtract = PPGFeatures(set.FS_GSR)

print(ppgExtract.extractTimeDomain(ppg.flatten()))

print(ppgExtract.extractFrequencyDomain(ppg.flatten()))

print(ppgExtract.extractNonLinear(ppg.flatten()))


#EDA
eda = data["GSR_GSR_Skin_Conductance_CAL"].iloc[0:900].values

edaExtract = EDAFeatures(set.FS_GSR)

print(edaExtract.extractMFCCFeatures(ppg.flatten()))

print(edaExtract.extractSCRFeatures(ppg.flatten()))
