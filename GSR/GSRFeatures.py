from Libs.MFCC.base import mfcc
from Libs.Utils import butterBandpassFilter
import numpy as np
import biosppy
from scipy import signal
import nolds
from Libs.cvxEDA.cvxEDA import cvxEDA
from scipy import interpolate


class PPGFeatures:
    def __init__(self, fs=1000):
        self.fs = fs

    def extractTimeDomain(self, x):
        '''
        :param x: raw ppg
        :return: time domain features of heart beat
        '''
        try:
            hb = biosppy.signals.bvp.bvp(x, sampling_rate=self.fs, show=False)[4]
            hb_mean = np.mean(hb)
            hb_max = np.max(hb)
            hb_min = np.min(hb)
            hb_std = np.std(hb)

            return np.array([hb_mean, hb_max, hb_min, hb_std])
        except:
            return np.array([])

    def extractFrequencyDomain(self, x):
        '''
        :param x: raw PPG
        :return:
        '''

        fbands = {'ulf': (0.00, 0.01), 'vlf': (0.01, 0.05), 'lf': (0.05, 0.15), 'hf': (0.15, 0.5)}
        onsets = biosppy.signals.bvp.bvp(x, sampling_rate=self.fs, show=False)[2]
        onsets_diff = np.insert(np.diff(onsets), 0, onsets[0]).astype(np.float)

        f, psd = signal.welch(onsets_diff, nperseg=4, fs=1, window="hamming")

        ulf = np.sum(psd[(f >= fbands["ulf"][0]) & (f < fbands["ulf"][1])])
        vlf = np.sum(psd[(f >= fbands["vlf"][0]) & (f < fbands["vlf"][1])])
        lf = np.sum(psd[(f >= fbands["lf"][0]) & (f < fbands["lf"][1])])
        hf = np.sum(psd[(f >= fbands["hf"][0]) & (f < fbands["hf"][1])])

        ulf_u = lf / (ulf + vlf + lf + hf)
        vlf_u = lf / (ulf + vlf + lf + hf)
        lf_u = lf / (lf + hf)
        hf_u = hf / (lf + hf)

        ratio = lf_u / hf_u

        return ulf_u, vlf_u, lf_u, hf_u, ratio

    def extractNonLinear(self, x):
        '''
        :param x: raw respiration data
        :return: zeros-crossing features (mean, min, max) and respiration rate (mean, min, max, vector) and nonlinear
        '''
        onsets = biosppy.signals.bvp.bvp(x, sampling_rate=self.fs, show=False)[2]
        onsets_diff = np.insert(np.diff(onsets), 0, onsets[0]).astype(np.float)

        # interpolate zeros_diff
        f = interpolate.interp1d(np.arange(0, len(onsets_diff)), onsets_diff)
        xnew = np.arange(0, len(onsets_diff) - 1, 0.5)
        onsets_diff_new = f(xnew)

        # nonlinear
        sample_ent = nolds.sampen(onsets_diff_new, emb_dim=1)


        lypanov_exp = nolds.lyap_e(onsets_diff_new, emb_dim=2, matrix_dim=2)[0]

        return np.array([sample_ent, lypanov_exp])


class EDAFeatures:

    def __init__(self, fs=1000):
        self.fs = fs

    def extractMFCCFeatures(self, x, winlen=2.0, stride=0.5, min_len=465):
        '''
        Compute melch frequency spectrum of EDA
        :param x: input signal
        :param winlen: windows length for framming. The default is 2.0 sec since the gradual changes in principle EDA towards stimulus is between 1.0 and 3.0 secs
        :param stride: string length for framming. The default is 0.5 sec
        :return: normalized melc coefficient
        '''
        x_len = len(x)
        len_diff = min_len - x_len
        if len_diff > 0:
            x = np.append(x, x[x_len - len_diff:])
        # print(x.shape)
        filtered = butterBandpassFilter(x, lowcut=0.03, highcut=5., order=4, fs=self.fs)
        melcfet = mfcc(filtered, samplerate=self.fs, winlen=winlen, winstep=stride)
        melcfet -= (np.mean(melcfet, axis=0) + 1e-18)
        # print(np.squeeze(np.squeeze(melcfet)).flatten().shape)
        return np.squeeze(np.squeeze(melcfet)).flatten()

    def extractCVXEDA(self, x):
        if np.std(x) < 1e-5:
            return np.array([])
        try:
            filtered = butterBandpassFilter(x, lowcut=0.03, highcut=5., order=4, fs=self.fs)
            yn = (filtered - filtered.mean()) / filtered.std()
            [r, p, t, l, d, _, _] = cvxEDA(yn, 1. / self.fs)

            # r features
            r_mean = np.mean(r)
            r_std = np.std(r)
            r_max = np.max(r)
            r_min = np.min(r)

            # p features
            p_mean = np.mean(p)
            p_std = np.std(p)
            p_max = np.max(p)
            p_min = np.min(p)

            # t features
            t_mean = np.mean(t)
            t_std = np.std(t)
            t_max = np.max(t)
            t_min = np.min(t)

            # l features
            l_mean = np.mean(l)
            l_std = np.std(l)
            l_max = np.max(l)
            l_min = np.min(l)

            # l features
            d_mean = np.mean(d)
            d_std = np.std(d)
            d_max = np.max(d)
            d_min = np.min(d)

            return np.array(
                [r_mean, r_std, r_max, r_min, p_mean, p_std, p_max, p_min, t_mean, t_std, t_max, t_min, l_mean, l_std,
                 l_max, l_min, d_mean, d_std, d_max, d_min])
        except:
            return np.array([])

    def extractSCRFeatures(self, x):
        '''
        :param x: input signal
        :return: scr onset indices, scr peak indices, scr amplitude
        '''
        try:
            _, _, onsets, peaks, amplitude = biosppy.eda.eda(x, sampling_rate=self.fs, show=False)
            onsets_diff = np.insert(np.diff(onsets), 0, onsets[0]).astype(np.float)
            peaks_diff = np.insert(np.diff(peaks), 0, peaks[0]).astype(np.float)

            # onsets features
            onsets_max = np.max(onsets_diff)
            onsets_min = np.min(onsets_diff)
            onsets_mean = np.mean(onsets_diff)
            onsets_std = np.std(onsets_diff)

            # peaks features
            peaks_max = np.max(peaks_diff)
            peaks_min = np.min(peaks_diff)
            peaks_mean = np.mean(peaks_diff)
            peaks_std = np.std(peaks_diff)

            # amplitude features
            amplitude_max = np.max(amplitude)
            amplitude_min = np.min(amplitude)
            amplitude_mean = np.mean(amplitude)
            amplitude_std = np.std(amplitude)
            return np.concatenate([np.array(
                [onsets_max, onsets_min, onsets_mean, onsets_std, peaks_max, peaks_min, peaks_mean, peaks_std,
                 amplitude_max, amplitude_min, amplitude_mean, amplitude_std])])
        except:
            return np.array([])
