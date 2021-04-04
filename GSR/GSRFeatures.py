from Libs.MFCC.base import mfcc
from scipy.stats import skew, kurtosis
import numpy as np
import biosppy
from biosppy import tools as st
from scipy import signal
import nolds
from Libs.cvxEDA.cvxEDA import cvxEDA
from scipy import interpolate
from Libs.Utils import rollingWindow


class PPGFeatures:
    def __init__(self, fs=1000):
        self.fs = fs


    def filterPPG(self, x):
        # filter signal
        # filtered = butterBandpassFilter(x, lowcut=.5, highcut=5., order=4, fs=self.fs)
        filtered, _, _ = st.filter_signal(signal=x,
                                          ftype='butter',
                                          band='bandpass',
                                          order=4,
                                          frequency=[1, 7],
                                          sampling_rate=self.fs)
        return filtered

    def extractTimeDomain(self, x):
        '''
        :param x: raw ppg
        :return: time domain features of heart beat
        '''
        try:
            onsets,  = biosppy.signals.bvp.find_onsets(x, sampling_rate=self.fs)
            # compute heart rate
            hb_index, hb = st.get_heart_rate(beats=onsets,
                                           sampling_rate=self.fs,
                                           smooth=True,
                                           size=3)
            hb_mean = np.mean(hb)
            hb_max = np.max(hb)
            hb_min = np.min(hb)
            hb_std = np.std(hb)
            hb_skew = skew(hb)
            hb_kurt = kurtosis(hb)

            return np.array([hb_mean, hb_max, hb_min, hb_std, hb_skew, hb_kurt])
        except:
            return np.array([])

    def extractFrequencyDomain(self, x):
        '''
        :param x: raw PPG
        :return:
        '''

        fbands = {'ulf': (0.00, 0.01), 'vlf': (0.0, 0.04), 'lf': (0.04, 0.15), 'hf': (0.15, 0.5)}
        onsets,  = biosppy.signals.bvp.find_onsets(x, sampling_rate=self.fs)

        onsets_diff = np.insert(np.diff(onsets), 0, onsets[0]).astype(np.float)

        f, psd = signal.welch(onsets_diff, nperseg=12, fs=1, window="hamming")

        # ulf = np.sum(psd[(f >= fbands["ulf"][0]) & (f < fbands["ulf"][1])])
        # vlf = np.sum(psd[(f >= fbands["vlf"][0]) & (f < fbands["vlf"][1])])
        lf = np.sum(psd[(f >= fbands["lf"][0]) & (f < fbands["lf"][1])])
        hf = np.sum(psd[(f >= fbands["hf"][0]) & (f < fbands["hf"][1])])

        #normalize
        # ulf_u = ulf / (ulf + vlf + lf + hf)
        # vlf_u = vlf / (vlf + lf + hf)
        #norm LF = LF / (Total Power - VLF)
        lf_u = lf / (lf + hf)
        hf_u = hf / (lf + hf)

        ratio = lf_u / hf_u

        return lf_u, hf_u, ratio

    def extractNonLinear(self, x):
        '''
        :param x: raw PPG
        :return: zeros-crossing features (mean, min, max) and respiration rate (mean, min, max, vector) and nonlinear
        '''
        onsets,  = biosppy.signals.bvp.find_onsets(x, sampling_rate=self.fs)
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

    def filterEDA(self, x):
        filtered, _, _ = st.filter_signal(signal=x,
                                     ftype='butter',
                                     band='lowpass',
                                     order=2,
                                     frequency=15,
                                     sampling_rate=self.fs)
        sm_size = int(0.75 * self.fs)
        filtered, _ = st.smoother(signal=filtered,
                              kernel='boxzen',
                              size=sm_size,
                              mirror=True)

        return filtered

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
        melcfet = mfcc(x, samplerate=self.fs, winlen=winlen, winstep=stride)
        melcfet -= (np.mean(melcfet, axis=0) + 1e-18)
        # print(np.squeeze(np.squeeze(melcfet)).flatten().shape)
        mfcc_features = np.squeeze(np.squeeze(melcfet)).flatten()
        mfcc_mean = np.mean(mfcc_features)
        mfcc_std = np.mean(mfcc_features)
        mfcc_median = np.median(mfcc_features)
        mfcc_skew = skew(mfcc_features)
        mfcc_kurt = kurtosis(mfcc_features)
        return np.array([mfcc_mean, mfcc_std, mfcc_median, mfcc_skew, mfcc_kurt])


    def meanSquare(self, x, axis):
        return np.sqrt(np.average(np.power(x, 2), axis=axis))

    def maxPSD(self, x):
        psd = np.abs(np.fft.fft(x))**2
        return np.max(psd, -1)

    def power(self, x):
        F = np.fft.fft(x)
        P = F * np.conj(F)
        resp = np.sum(P, -1)

        return np.mean(resp.real)
    def extractCVXEDA(self, x, size = .5):
        if np.std(x) < 1e-5:
            return np.array([])
        try:

            yn = (x - x.mean()) / x.std()
            [r, p, t, l, d, _, _] = cvxEDA(yn, 1. / self.fs)
            n = int(size * self.fs)
            # r features
            r_rolling = rollingWindow(r, size=n)
            r_mean = np.mean(r_rolling, 1).flatten()
            r_std = np.std(r_rolling, 1).flatten()
            r_mean_square = self.meanSquare(r_rolling, 1).flatten()
            r_power = self.power(r_rolling).flatten()
            r_psd = self.maxPSD(r_rolling).flatten()
            r_skew = skew(r)
            r_kurt = kurtosis(r)


            # p features
            p_rolling = rollingWindow(p, size=n)
            p_mean = np.mean(p_rolling, 1).flatten()
            p_std = np.std(p_rolling, 1).flatten()
            p_mean_square = self.meanSquare(p_rolling, 1).flatten()
            p_power = self.power(p_rolling).flatten()
            p_psd = self.maxPSD(p_rolling).flatten()
            p_skew = skew(p)
            p_kurt = kurtosis(p)

            # t features
            t_rolling = rollingWindow(t, size=n)
            t_mean = np.mean(t_rolling, 1).flatten()
            t_std = np.std(t_rolling, 1).flatten()
            t_mean_square = self.meanSquare(t_rolling, 1).flatten()
            t_power = self.power(t_rolling).flatten()
            t_psd =  self.maxPSD(t_rolling).flatten()
            t_skew = skew(t)
            t_kurt = kurtosis(t)

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

            return np.concatenate(
                [r_mean, r_std, r_mean_square, r_power, r_psd,  p_mean, p_std, p_mean_square, p_power, p_psd, t_mean, t_std, t_mean_square, t_power, t_psd,  np.array([p_skew, p_kurt, t_skew, t_kurt,  r_skew, r_kurt, l_mean, l_std,
                 l_max, l_min, d_mean, d_std, d_max, d_min])])

            # return np.array(
            #     [r_mean, r_std, r_mean_square, r_power, r_psd,  p_mean, p_std, p_mean_square, p_power, p_psd, t_mean, t_std, t_mean_square, t_power, t_psd,  p_skew, p_kurt, t_skew, t_kurt,  r_skew, r_kurt, l_mean, l_std,
            #      l_max, l_min, d_mean, d_std, d_max, d_min])
        except:
            return np.array([])

    def extractSCRFeatures(self, x):
        '''
        :param x: input signal
        :return: scr onset indices, scr peak indices, scr amplitude
        '''
        try:
            #_, _, onsets, peaks, amplitude = biosppy.eda.eda(x, sampling_rate=self.fs, show=False)

            onsets, peaks, amplitude = biosppy.eda.basic_scr(signal=x,
                                          sampling_rate=self.fs)
            onsets_diff = np.insert(np.diff(onsets), 0, onsets[0]).astype(np.float)
            peaks_diff = np.insert(np.diff(peaks), 0, peaks[0]).astype(np.float)

            # onsets features
            onsets_skew = skew(onsets_diff)
            onsets_kurt = kurtosis(onsets_diff)
            onsets_mean = np.mean(onsets_diff)
            onsets_std = np.std(onsets_diff)

            # peaks features
            peaks_skew = skew(peaks_diff)
            peaks_kurt = kurtosis(peaks_diff)
            peaks_mean = np.mean(peaks_diff)
            peaks_std = np.std(peaks_diff)

            # amplitude features
            amplitude_skew = skew(amplitude)
            amplitude_kurt = kurtosis(amplitude)
            amplitude_mean = np.mean(amplitude)
            amplitude_std = np.std(amplitude)

            return np.concatenate([np.array(
                [onsets_skew, onsets_kurt, onsets_mean, onsets_std, peaks_skew, peaks_kurt, peaks_mean, peaks_std,
                 amplitude_skew, amplitude_kurt, amplitude_mean, amplitude_std])])
        except:
            return np.array([])
