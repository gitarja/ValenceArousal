from Libs.MFCC.base import mfcc
from Libs.Utils import butterBandpassFilter
import numpy as np
import biosppy
from scipy import signal
import nolds

class PPGFeatures:
    def __init__(self, fs=1000):
        self.fs = fs

    def extractTimeDomain(self, x):
        '''
        :param x: raw ppg
        :return: time domain features of heart beat
        '''
        hb = biosppy.signals.bvp.bvp(x, sampling_rate=self.fs, show=False)[4]
        hb_mean = np.mean(hb)
        hb_max = np.max(hb)
        hb_min = np.min(hb)
        hb_std = np.std(hb)

        return hb, hb_mean, hb_max, hb_min, hb_std


    def extractFrequencyDomain(self, x):
        '''
        :param x: raw PPG
        :return:
        '''


        fbands = {'ulf': (0.00, 0.01), 'vlf': (0.01, 0.05), 'lf': (0.05, 0.15), 'hf': (0.15, 0.5)}
        onsets = biosppy.signals.bvp.bvp(x, sampling_rate=self.fs, show=False)[2]
        onsets_diff = np.insert(np.diff(onsets), 0, onsets[0]).astype(np.float)

        f, psd = signal.welch(onsets_diff, nperseg=16, fs=1, window="hamming")

        ulf = np.sum(psd[(f>=fbands["ulf"][0]) & (f < fbands["ulf"][1])])
        vlf = np.sum(psd[(f>=fbands["vlf"][0]) & (f < fbands["vlf"][1])])
        lf = np.sum(psd[(f>=fbands["lf"][0]) & (f < fbands["lf"][1])])
        hf = np.sum(psd[(f>=fbands["hf"][0]) & (f < fbands["hf"][1])])

        lf_u = lf / (lf + hf)
        hf_u = hf / (lf + hf)

        ratio = lf_u / hf_u

        return ulf, vlf, lf_u, hf_u, ratio
    def extractNonLinear(self, x):
        '''
        :param x: raw respiration data
        :return: zeros-crossing features (mean, min, max) and respiration rate (mean, min, max, vector) and nonlinear
        '''
        onsets = biosppy.signals.bvp.bvp(x, sampling_rate=self.fs, show=False)[2]
        onsets_diff = np.insert(np.diff(onsets), 0, onsets[0]).astype(np.float)
        # nonlinear
        sample_ent = nolds.sampen(onsets_diff, emb_dim=1)
        lypanov_exp = nolds.lyap_e(onsets_diff, emb_dim=2, matrix_dim=2)[0]

        return  sample_ent, lypanov_exp




class EDAFeatures:

    def __init__(self, fs = 1000):
        self.fs = fs

    def extractMFCCFeatures(self, x, winlen=2.0, stride=0.5):
        '''
        Compute melch frequency spectrum of EDA
        :param x: input signal
        :param winlen: windows length for framming. The default is 2.0 sec since the gradual changes in principle EDA towards stimulus is between 1.0 and 3.0 secs
        :param stride: string length for framming. The default is 0.5 sec
        :return: normalized melc coefficient
        '''
        filtered = butterBandpassFilter(x, lowcut=0.03, highcut=5., order=4, fs=self.fs)
        melcfet = mfcc(filtered, samplerate=self.fs, winlen=winlen, winstep=stride)
        melcfet -= (np.mean(melcfet, axis=0) + 1e-18)
        return np.squeeze(np.squeeze(melcfet))

    def extractSCRFeatures(self, x):
        '''
        :param x: input signal
        :return: scr onset indices, scr peak indices, scr amplitude
        '''
        _, _, onsets, peaks, amplitude = biosppy.eda.eda(x, sampling_rate=self.fs, show=False)
        onsets_diff = np.insert(np.diff(onsets), 0, onsets[0]).astype(np.float)
        peaks_diff = np.insert(np.diff(peaks), 0, peaks[0]).astype(np.float)

        #onsets features
        onsets_max = np.max(onsets_diff)
        onsets_min = np.min(onsets_diff)
        onsets_mean = np.mean(onsets_diff)
        onsets_std = np.std(onsets_diff)

        # peaks features
        peaks_max = np.max(peaks_diff)
        peaks_min = np.min(peaks_diff)
        peaks_mean = np.mean(peaks_diff)
        peaks_std = np.std(peaks_diff)
        return onsets_max, onsets_min, onsets_mean, onsets_std, peaks_max, peaks_min, peaks_mean, peaks_std, amplitude
