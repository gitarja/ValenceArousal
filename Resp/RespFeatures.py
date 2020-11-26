from scipy.stats import skew, kurtosis
import numpy as np
from scipy import signal
from scipy import interpolate
import nolds
from biosppy import tools as st

class RespFeatures:

    def __init__(self, fs):
        self.fs = fs

    def extracRate(self, x, fs):
        # compute zero crossings
        zeros, = st.zero_cross(signal=x, detrend=True)
        beats = zeros[::2]

        if len(beats) < 2:
            rate_idx = []
            rate = []
        else:
            # compute respiration rate
            rate_idx = beats[1:]
            rate = fs * (1. / np.diff(beats))

            # physiological limits
            indx = np.nonzero(rate <= 0.35)
            rate_idx = rate_idx[indx]
            rate = rate[indx]

            # smooth with moving average
            size = 3

            rate, _ = st.smoother(signal=rate,
                                  kernel='boxcar',
                                  size=size,
                                  mirror=True)

            return zeros, rate
    def filterResp(self, x):
        # fc = 2.  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
        # b = 0.08  # 120 taps
        # N = int(np.ceil((4 / b)))
        # if not N % 2: N += 1  # Make sure that N is odd.
        # n = np.arange(N)
        #
        # # Compute sinc filter.
        # h = np.sinc(2 * fc * (n - (N - 1) / 2))
        #
        # # Compute Blackman window.
        # w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
        #     0.08 * np.cos(4 * np.pi * n / (N - 1))
        #
        # # Multiply sinc filter by window.
        # h = h * w
        #
        # # Normalize to get unity gain.
        # h = h / np.sum(h)
        #
        # return np.convolve(x, h, mode="same")

        filtered, _, _ = st.filter_signal(signal=x,
                                          ftype='butter',
                                          band='bandpass',
                                          order=2,
                                          frequency=[0.1, 0.35],
                                          sampling_rate=self.fs)

        return filtered

    def extractTimeDomain(self, x):
        '''
        :param x: raw respiration data
        :return: zeros-crossing features (mean, min, max) and respiration rate (mean, min, max, vector) and nonlinear
        '''
        try:
            resp_rate = self.extracRate(x, self.fs)[1]
            # resp features
            resp_mean = np.mean(resp_rate)
            resp_max = np.max(resp_rate)
            resp_min = np.min(resp_rate)
            resp_std = np.std(resp_rate)
            resp_skew = skew(resp_rate)
            resp_kurt = kurtosis(resp_rate)
            return np.array([resp_mean, resp_max, resp_min, resp_std, resp_skew, resp_kurt])
        except:
            return np.array([])

    def extractNonLinear(self, x):
        '''
        :param x: raw respiration data
        :return: zeros-crossing features (mean, min, max) and respiration rate (mean, min, max, vector) and nonlinear
        '''
        zeros = self.extracRate(x, self.fs)[0]
        zeros_diff = np.insert(np.diff(zeros), 0, zeros[0]).astype(np.float)

        # interpolate zeros_diff
        f = interpolate.interp1d(np.arange(0, len(zeros_diff)), zeros_diff)
        xnew = np.arange(0, len(zeros_diff) - 1, 0.5)
        zeros_diff_new = f(xnew)
        # nonlinear
        sample_ent = nolds.sampen(zeros_diff_new, emb_dim=1)
        lypanov_exp = nolds.lyap_e(zeros_diff_new, emb_dim=2, matrix_dim=2)[0]

        return np.array([sample_ent, lypanov_exp])

    def extractFrequencyDomain(self, x):
        '''
        :param x: raw respiration data
        :return:
        '''
        fbands = {'ulf': (0.00, 0.01), 'vlf': (0.0, 0.1), 'lf': (0.1, 0.2), 'hf': (0.15, 0.5)}
        zeros = self.extracRate(x, self.fs)[0]
        zeros_diff = np.insert(np.diff(zeros), 0, zeros[0]).astype(np.float)

        f, psd = signal.welch(zeros_diff, nperseg=12, fs=1, window="hamming")

        # ulf = np.sum(psd[(f >= fbands["ulf"][0]) & (f < fbands["ulf"][1])])
        # vlf = np.sum(psd[(f >= fbands["vlf"][0]) & (f < fbands["vlf"][1])])
        lf = np.sum(psd[(f >= fbands["lf"][0]) & (f < fbands["lf"][1])])
        hf = np.sum(psd[(f >= fbands["hf"][0]) & (f < fbands["hf"][1])])

        # ulf_u = ulf / (ulf + vlf + lf + hf)
        # vlf_u = vlf / (vlf + lf + hf)
        lf_u = lf / (lf + hf)
        hf_u = hf / (lf + hf)

        ratio = lf_u / hf_u

        return np.array([lf_u, hf_u, ratio])
