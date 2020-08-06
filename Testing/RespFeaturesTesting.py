from Resp.RespFeatures import RespFeatures
from Conf import Settings as set
import pandas as pd

data = pd.read_csv("..\\Data\\Dummy\\Resp_sample.csv", header=[0,1])

resp = data["RESP_ECG_RESP_24BIT_CAL"].values

respExtract = RespFeatures(set.FS_RESP)

print(respExtract.extractTimeDomain(resp.flatten()))

print(respExtract.extractFrequencyDomain(resp.flatten()))

print(respExtract.extractNonLinear(resp.flatten()))