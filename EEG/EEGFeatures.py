from Libs import pyeeg
import numpy as np
import pywt
from Libs.Utils import butterBandpassFilter, avgSlidingWindow
from biosppy import tools as st
from EEG.SpaceLapFilter import SpaceLapFilter
import biosppy
from Libs.Utils import rollingWindow


class EEGFeatures:

    def __init__(self, fs=1000):
        self.fs = fs

    def filterEEG(self, x):
        # SpaceLapFilter
        space_lap_filter = SpaceLapFilter()
        filtered = space_lap_filter.FilterEEG(x, mode=8)

        # high pass filter
        b, a = st.get_filter(ftype='butter',
                             band='highpass',
                             order=8,
                             frequency=4,
                             sampling_rate=self.fs)

        filtered, _ = st._filter_signal(b, a, signal=filtered, check_phase=True, axis=0)

        # low pass filter
        b, a = st.get_filter(ftype='butter',
                             band='lowpass',
                             order=16,
                             frequency=100,
                             sampling_rate=self.fs)

        filtered, _ = st._filter_signal(b, a, signal=filtered, check_phase=True, axis=0)

        # smoothed
        for i in range(filtered.shape[1]):
            filtered[:, i] = avgSlidingWindow(filtered[:, i], n=50)

        return filtered

    def preProcessing(self, x, n=50):
        lc = 4
        hc = 55
        filtered = butterBandpassFilter(x, lowcut=lc, highcut=hc, fs=self.fs)
        smoothed = avgSlidingWindow(filtered, n=n)

        return smoothed

    def extractThetaAlphaBeta(self, x):
        '''
        ref:https://www.journals.elsevier.com/clinical-neurophysiology/view-for-free/guidelines-of-the-ifcn-2nd-ed-published-1999
        :param x: eeg signal
        :return: theta, alpha, and beta of the signal
        '''
        filtered = self.preProcessing(x)
        theta = butterBandpassFilter(filtered, lowcut=4, highcut=8, fs=self.fs, order=2)
        alpha_low = butterBandpassFilter(filtered, lowcut=8, highcut=10, fs=self.fs, order=2)
        alpha_high = butterBandpassFilter(filtered, lowcut=10, highcut=14, fs=self.fs, order=2)
        beta = butterBandpassFilter(filtered, lowcut=14, highcut=25, fs=self.fs, order=2)
        gamma_low = butterBandpassFilter(filtered, lowcut=25, highcut=40, fs=self.fs, order=2)
        gamma_high = butterBandpassFilter(filtered, lowcut=40, highcut=100, fs=self.fs, order=2)

        return theta, alpha_low, alpha_high, beta, gamma_low, gamma_high

    def mean(self, x):
        return np.mean(x)

    def std(self, x):
        return np.std(x)

    def meanSquare(self, x, axis):
        return np.sqrt(np.average(np.power(x, 2), axis=axis))

    def hjort(self, x):
        hjrots = []
        for i in range(len(x)):
            hjrot = pyeeg.hjorth(x[i])
            hjrots.append(hjrot[1])
        return np.mean(np.array(hjrots))

    def maxPSD(self, x):
        psd = np.abs(np.fft.fft(x)) ** 2
        return np.max(psd, -1)

    def power(self, x):
        F = np.fft.fft(x)
        P = F * np.conj(F)
        resp = np.sum(P, -1)

        return np.mean(resp.real)

    def extractTimeDomainFeatures(self, x):
        '''
        :param x: eeg signal
        :return: time-domain features of theta, alpha, and beta
        '''
        theta, alpha_low, alpha_high, beta, gamma_low, gamma_high = self.extractThetaAlphaBeta(x)

        features_t = self.extractTimeDomain(theta)
        features_al = self.extractTimeDomain(alpha_low)
        features_ah = self.extractTimeDomain(alpha_high)
        features_b = self.extractTimeDomain(beta)
        features_gl = self.extractTimeDomain(gamma_low)
        features_gh = self.extractTimeDomain(gamma_high)

        return np.concatenate([features_t, features_al, features_ah, features_b, features_gl, features_gh])

    def extractTimeDomainAll(self, x):
        features = []
        for i in range(x.shape[1]):
            features.append(self.extractTimeDomainFeatures(x[:, i]))

        return np.concatenate(features)

    def extractFreqTimeDomainAll(self, x):
        features = []
        for i in range(x.shape[1]):
            features.append(self.extractFreqTimeFeatures(x[:, i]))

        return np.concatenate(features)

    def extractFrequencyDomainAll(self, x):
        features = []
        for i in range(x.shape[1]):
            features.append(self.extractFrequencyDomainFeatures(x[:, i]))

        return np.concatenate(features)

    def extractTimeDomain(self, x):
        '''
        :param x: a signal
        :return: time-domain features
        '''
        n = 4000
        x = rollingWindow(x, size=n)
        # m = np.mean(np.mean(x, 1).flatten())
        # std = np.mean(np.mean(x, 1).flatten())
        rms = np.mean(self.meanSquare(x, 1).flatten())
        hjrot = self.hjort(x)
        maxPsd = np.mean(self.maxPSD(x))
        power = self.power(x)
        return np.array([rms, hjrot, maxPsd, power])

    def bandPower(self, x, bands, fs):
        return pyeeg.bin_power(x, bands, fs)

    def extractFrequencyDomainFeatures(self, x, bands=[4, 8, 10, 14, 25, 40, 100]):
        '''
        ref: https://www.journals.elsevier.com/clinical-neurophysiology/view-for-free/guidelines-of-the-ifcn-2nd-ed-published-1999
        :param x: eeg signal
        :param bands: ranges of frequencies (theta: 4-8, alpha:8-14, beta: 14-25)
        :return: power spectrum of theta, alpha, and beta
        '''
        filtered = self.preProcessing(x)
        _, pwr_t = self.bandPower(filtered, bands, self.fs)

        return np.array(pwr_t)

    def extractPLFFeatures(self, x):
        _, _, PLF = biosppy.eeg.get_plf_features(x, sampling_rate=self.fs)
        PLF_mean = np.mean(PLF, 0)
        PLF_std = np.std(PLF, 0)
        return np.concatenate([PLF_mean, PLF_std])

    def extractPowerFeatures(self, x):
        _, theta, alpha_low, alpha_high, beta, gamma = biosppy.eeg.get_power_features(x, sampling_rate=self.fs)
        theta_mean = np.mean(theta, 0)
        theta_std = np.std(theta, 0)
        alpha_low_mean = np.mean(alpha_low, 0)
        alpha_low_std = np.std(alpha_low, 0)
        alpha_high_mean = np.mean(alpha_high, 0)
        alpha_high_std = np.std(alpha_high, 0)
        beta_mean = np.mean(beta, 0)
        beta_std = np.std(beta, 0)
        gamma_mean = np.mean(gamma, 0)
        gamma_std = np.std(gamma, 0)
        features = np.concatenate(
            [theta_mean, theta_std, alpha_low_mean, alpha_low_std, alpha_high_mean, alpha_high_std, beta_mean, beta_std,
             gamma_mean, gamma_std])
        return features

    def extractFreqTimeFeatures(self, x, level=8):
        '''
        compute the frequency-time domain features using bior3.3 wavelet
        :param x: eeg signal
        :param level: frequency-time domain features of theta, alpha, and beta
        :return:
        '''

        theta, alpha_low, alpha_high, beta, gamma_low, gamma_high = self.extractThetaAlphaBeta(x)

        coeffs_t = pywt.wavedecn(theta, "bior3.3", level=level)
        coeffs_al = pywt.wavedecn(alpha_low, "bior3.3", level=level)
        coeffs_ah = pywt.wavedecn(alpha_high, "bior3.3", level=level)
        coeffs_b = pywt.wavedecn(beta, "bior3.3", level=level)
        coeffs_gl = pywt.wavedecn(gamma_low, "bior3.3", level=level)
        coeffs_gh = pywt.wavedecn(gamma_high, "bior3.3", level=level)

        return np.concatenate([coeffs_t[level]['d'], coeffs_al[level]['d'], coeffs_ah[level]['d'], coeffs_b[level]['d'],
                               coeffs_gl[level]['d'], coeffs_gh[level]['d']])
