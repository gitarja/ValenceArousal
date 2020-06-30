import pyeeg
import numpy as np
import pywt
from Libs.Utils import Utils

class EEGFeatures:

    def __init__(self, fs = 1000):
        self.utils = Utils()
        self.fs = fs

    def preProcessing(self, x, n=50):
        lc = 4
        hc = 55
        filtered = self.utils.butterBandpassFilter(x, lowcut=lc, highcut=hc, fs=self.fs)
        smoothed = self.utils.avgSlidingWindow(filtered, n=50)

        return smoothed


    def extractThetaAlphaBeta(self, x):
        '''
        :param x: eeg signal
        :return: theta, alpha, and beta of the signal
        '''
        theta = self.utils.butterBandpassFilter(x, lowcut=4, highcut=7, fs=self.fs)
        alpha = self.utils.butterBandpassFilter(x, lowcut=8, highcut=15, fs=self.fs)
        beta = self.utils.butterBandpassFilter(x, lowcut=16, highcut=31, fs=self.fs)

        return theta, alpha, beta

    def mean(self, x):
        return np.mean(x)

    def std(self, x):
        return np.std(x)

    def meanSquare(self, x):
        return np.sqrt(np.average(np.power(x, 2)))

    def hjort(self, x):
        return pyeeg.hjorth(x)

    def maxPSD(self, x):
        psd = np.abs(np.fft.fft(x))**2
        return np.max(psd)

    def power(self, x):
        F = np.fft.fft(x)
        P = F * np.conj(F)
        resp = np.sum(P)

        return resp.real

    def timeDomainFeatures(self, x):
        '''
        :param x: eeg signal
        :return: time-domain features of theta, alpha, and beta
        '''
        t, a, b = self.extractThetaAlphaBeta(x)

        features_t = self.timeDomain(t)
        features_a = self.timeDomain(a)
        features_b = self.timeDomain(b)

        return features_t, features_a, features_b

    def timeDomain(self, x):
        '''
        :param x: a signal
        :return: time-domain features
        '''
        m = self.mean(x)
        std = self.std(x)
        rms = self.meanSquare(x)
        hjrot = self.hjort(x)
        maxPsd = self.maxPSD(x)
        power = self.power(x)
        return np.concatenate([m, std, rms, hjrot[0], hjrot[1], maxPsd, power])


    def bandPower(self, x, bands, fs):
       return pyeeg.bin_power(x, bands, fs)

    def frequencyDomainFeatures(self, x, bands):
        '''
        :param x: eeg signal
        :param bands: ranges of frequencies
        :return: power spectrum of theta, alpha, and beta
        '''
        t,a,b = self.extractThetaAlphaBeta(x)

        _, pwr_t = self.bandPower(t, bands, self.fs)
        _, pwr_a = self.bandPower(a, bands, self.fs)
        _, pwr_b = self.bandPower(b, bands, self.fs)

        return pwr_t, pwr_a, pwr_b

    def freqTimeFeatures(self, x, level=8):
        '''
        compute the frequency-time domain features using bior3.3 wavelet
        :param x: eeg signal
        :param level: frequency-time domain features of theta, alpha, and beta
        :return:
        '''

        t, a, b = self.extractThetaAlphaBeta(x)

        coeffs_t = pywt.wavedecn(t, "bior3.3", level=level)
        coeffs_a = pywt.wavedecn(a, "bior3.3", level=level)
        coeffs_b = pywt.wavedecn(b, "bior3.3", level=level)

        return coeffs_t[level]['d'], coeffs_a[level]['d'], coeffs_b[level]['d']





