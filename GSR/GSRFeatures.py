from Libs.MFCC.base import mfcc
import numpy as np
from biosppy.signals.eda import eda
import biosppy
class GSRFeatures:

    def __init__(self, fs = 1000):
        self.fs = fs

    def computeHeartBeat(self, x, fs):
        ts, hb = biosppy.signals.bvp.bvp(x, sampling_rate=fs, show=False)[3:]
        return ts, hb

    def mfccFeatures(self, x, winlen=2.0, stride=0.5):
        '''
        :param x: input signal
        :param winlen: windows length for framming. The default is 2.0 sec since the gradual changes in principle EDA towards stimulus is between 1.0 and 3.0 secs
        :param stride: string length for framming. The default is 0.5 sec
        :return: normalized melc coefficient
        '''
        melcfet = mfcc(x, samplerate=self.fs, winlen=winlen, winstep=stride)
        melcfet -= (np.mean(melcfet, axis=0) + 1e-18)
        return np.squeeze(np.squeeze(melcfet))

    def scrFeatures(self, x):
        '''
        :param x: input signal
        :return: scr onset indices, scr peak indices, scr amplitude
        '''
        _, _, onset, peaks, amplitude = eda(x, sampling_rate=self.fs, show=False)

        return onset, peaks, amplitude
