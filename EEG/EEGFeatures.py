from Libs import pyeeg
import numpy as np
import pywt
from Libs.Utils import butterBandpassFilter, avgSlidingWindow

class EEGFeatures:

    def __init__(self, fs = 1000):
        self.fs = fs

    def preProcessing(self, x, n=50):
        lc = 4
        hc = 55
        filtered = butterBandpassFilter(x, lowcut=lc, highcut=hc, fs=self.fs)
        smoothed = avgSlidingWindow(filtered, n=n)

        return smoothed


    def extractThetaAlphaBeta(self, x):
        '''
        :param x: eeg signal
        :return: theta, alpha, and beta of the signal
        '''
        filtered = self.preProcessing(x)
        theta = butterBandpassFilter(filtered, lowcut=4, highcut=7, fs=self.fs, order=2)
        alpha = butterBandpassFilter(filtered, lowcut=8, highcut=15, fs=self.fs, order=2)
        beta = butterBandpassFilter(filtered, lowcut=16, highcut=31, fs=self.fs, order=2)

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

    def extractTimeDomainFeatures(self, x):
        '''
        :param x: eeg signal
        :return: time-domain features of theta, alpha, and beta
        '''
        t, a, b = self.extractThetaAlphaBeta(x)

        features_t = self.extractTimeDomain(t)
        features_a = self.extractTimeDomain(a)
        features_b = self.extractTimeDomain(b)

        return np.concatenate([features_t, features_a, features_b])

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
        m = self.mean(x)
        std = self.std(x)
        rms = self.meanSquare(x)
        hjrot = self.hjort(x)
        maxPsd = self.maxPSD(x)
        power = self.power(x)
        return np.array([m, std, rms, hjrot[0], hjrot[1], maxPsd, power])


    def bandPower(self, x, bands, fs):
       return pyeeg.bin_power(x, bands, fs)

    def extractFrequencyDomainFeatures(self, x, bands=[4, 8, 16, 32]):
        '''
        :param x: eeg signal
        :param bands: ranges of frequencies (theta: 4-7, alpha:8-15, beta: 16-32)
        :return: power spectrum of theta, alpha, and beta
        '''
        filtered = self.preProcessing(x)
        _, pwr_t = self.bandPower(filtered, bands, self.fs)


        return np.array(pwr_t)

    def extractFreqTimeFeatures(self, x, level=8):
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





