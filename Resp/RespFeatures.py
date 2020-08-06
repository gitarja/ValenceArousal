import biosppy
import numpy as np
from Libs.Utils import butterBandpassFilter
from scipy import signal
import nolds


class RespFeatures:

    def __init__(self, fs):
        self.fs = fs

    def extractTimeDomain(self, x):
        '''
        :param x: raw respiration data
        :return: zeros-crossing features (mean, min, max) and respiration rate (mean, min, max, vector) and nonlinear
        '''
        resp_rate = biosppy.resp.resp(signal=x, sampling_rate=self.fs, show=False)[4]

        # resp features
        resp_mean = np.mean(resp_rate)
        resp_max = np.max(resp_rate)
        resp_min = np.min(resp_rate)


        return resp_mean, resp_max, resp_min, resp_rate

    def extractNonLinear(self, x):
        '''
        :param x: raw respiration data
        :return: zeros-crossing features (mean, min, max) and respiration rate (mean, min, max, vector) and nonlinear
        '''
        zeros = biosppy.resp.resp(signal=x, sampling_rate=self.fs, show=False)[2]
        zeros_diff = np.insert(np.diff(zeros), 0, zeros[0]).astype(np.float)
        # nonlinear
        sample_ent = nolds.sampen(zeros_diff, emb_dim=1)
        lypanov_exp = nolds.lyap_e(zeros_diff, emb_dim=2, matrix_dim=2)[0]

        return  sample_ent, lypanov_exp

    def extractFrequencyDomain(self, x):
        '''
        :param x: raw respiration data
        :return:
        '''
        fbands = {'ulf': (0.00, 0.01), 'vlf': (0.01, 0.05), 'lf': (0.05, 0.15), 'hf': (0.15, 0.5)}
        zeros = biosppy.resp.resp(signal=x, sampling_rate=self.fs, show=False)[2]
        zeros_diff = np.insert(np.diff(zeros), 0, zeros[0]).astype(np.float)

        f, psd = signal.welch(zeros_diff, nperseg=16, fs=1, window="hamming")

        ulf = np.sum(psd[(f >= fbands["ulf"][0]) & (f < fbands["ulf"][1])])
        vlf = np.sum(psd[(f >= fbands["vlf"][0]) & (f < fbands["vlf"][1])])
        lf = np.sum(psd[(f >= fbands["lf"][0]) & (f < fbands["lf"][1])])
        hf = np.sum(psd[(f >= fbands["hf"][0]) & (f < fbands["hf"][1])])

        lf_u = lf / (lf + hf)
        hf_u = hf / (lf + hf)

        ratio = lf_u / hf_u

        return ulf, vlf, lf_u, hf_u, ratio
