import biosppy
import nolds
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nn
import pyhrv.tools as tools
import pyhrv
from biosppy import utils
import numpy as np
from biosppy import tools as st
import pywt


class ECGFeatures:

    def __init__(self, fs):
        self.fs = fs

    def filterECG(self, x):
        order = int(0.3 * self.fs)
        filtered, _, _ = st.filter_signal(signal=x,
                                          ftype='FIR',
                                          band='bandpass',
                                          order=order,
                                          frequency=[3, 45],
                                          sampling_rate=self.fs)

        return filtered

    def waveDriftFilter(self, signal, n=9):
        '''
        :param signal: signal input
        :param n: level of decomposition
        :return: signal - baseline
        The signal is decomposed into 9 level and only the last coeffictient [1] is used to reconstruct the baseline
        To remove baseline wondering the baseline is substracted from the input signal
        '''
        waveletName = "bior1.5"
        coeffs = pywt.wavedecn(signal, waveletName, level=n)
        for i in range(2, len(coeffs), 1):
            coeffs[i] = self.ignoreCoefficient(coeffs[i])
        baseline = pywt.waverecn(coeffs, waveletName)
        filtered = signal - baseline[:len(signal)]
        return filtered

    def ignoreCoefficient(self, coeff):
        coeff = {k: np.zeros_like(v) for k, v in coeff.items()}
        return coeff

    def extractRR(self, x):

        r = biosppy.signals.ecg.ecg(x, sampling_rate=self.fs, show=False)[2]
        r = r.astype(float)
        # Compute NNI or RR
        nni = tools.nn_intervals(r)

        return nni * 4

    def computeHeartBeat(self, x):
        ts, hb = biosppy.signals.ecg.ecg(x, sampling_rate=self.fs, show=False)[5:]
        return ts, hb

    def extractTimeDomain(self, x):
        try:
            nni = self.extractRR(x)
            nniParams = td.nni_parameters(nni=nni)
            nniSD = td.sdnn(nni=nni)
            nniDiff = td.nni_differences_parameters(nni=nni)
            nniDiffRM = td.rmssd(nni=nni)
            nniDiffSD = td.sdsd(nni=nni)
            hrParams = td.hr_parameters(nni=nni)
            nn20 = td.nn20(nni=nni)
            nn30 = td.nnXX(nni=nni, threshold=30)
            nn50 = td.nn50(nni=nni)
            # return np.array([nniParams["nni_mean"], nniParams["nni_counter"], nniSD["sdnn"],
            #        nniDiff["nni_diff_mean"], nniDiffRM["rmssd"], nniDiffSD["sdsd"],
            #        hrParams["hr_mean"], hrParams["hr_std"]])

            return np.array([nniParams["nni_mean"], nniParams["nni_counter"], nniSD["sdnn"],
                             nniDiff["nni_diff_mean"], nniDiffRM["rmssd"], nniDiffSD["sdsd"],
                             hrParams["hr_mean"], hrParams["hr_std"], hrParams["hr_max"] - hrParams["hr_min"],
                             nn20["nn20"], nn20["pnn20"], nn30["nn30"], nn30["pnn30"], nn50["nn50"], nn50["pnn50"]])

        except:
            return np.array([])

    def extractFrequencyDomain(self, x):
        try:
            nni = self.extractRR(x)
            # the bands was decided by refering to Revealing Real-Time Emotional Responses
            psd = fd.welch_psd(nni=nni, show=False,
                               fbands={'ulf': (0.00, 0.01), 'vlf': (0.01, 0.05), 'lf': (0.05, 0.15), 'hf': (0.15, 0.5)},
                               nfft=2 ** 12, legend=False, mode="dev")[0]
            features = np.concatenate([np.array(psd["fft_peak"]), np.array(psd["fft_abs"]), np.array(psd["fft_rel"]), np.array(psd["fft_log"]), np.array([psd["fft_norm"][0], psd["fft_norm"][1], psd["fft_ratio"]])])
            return features

        except:
            return np.array([])

    def extractNonLinearDomain(self, x):
        try:
            nni = self.extractRR(x)
            sampEntro = nn.sample_entropy(nni=nni, dim=2) #change dim from 1 to 2
            lyapEx = self.lyapunov_exponent(nni=nni, emb_dim=3, matrix_dim=2)
            pointCare = nn.poincare(nni=nni, show=False)
            # hrust = nolds.hurst_rs(nni)
            corrDim = nolds.corr_dim(nni, emb_dim=3)
            # dfa = nolds.dfa(nni)
            # return np.array([sampEntro["sampen"], lyapEx["lyapex"]])
            return np.array([sampEntro["sampen"], lyapEx["lyapex"], pointCare["sd1"], pointCare["sd2"], pointCare["sd_ratio"], pointCare["ellipse_area"],  corrDim])

        except:
            return np.array([])


    def lyapunov_exponent(self, nni=None, rpeaks=None, emb_dim=10, matrix_dim=4):
            """
            Computes Lyapunov Exponent for of the NNi series
            The first LE is considered as the instantaneous dominant of LE
            Recommendations for parameter settings by Eckmann et al.:
                - long recording time improves accuracy, small tau does not
                - Use large values for emb_dim
                - Matrix_dim should be ‘somewhat larger than the expected number of positive Lyapunov exponents’
                - Min_nb = min(2 * matrix_dim, matrix_dim + 4)
            :param nni:
            :param rpeaks:
            :param emb_dim:
            :param matrix_dim:expected dimension of lyapunov exponential
            :return: the first LE
            """
            min_nb = min(2 * matrix_dim, matrix_dim + 4)

            # Check input values
            nn = pyhrv.utils.check_input(nni, rpeaks)

            # compute Lyapunov Exponential
            lyapex = nolds.lyap_e(data=nn, emb_dim=emb_dim, matrix_dim=matrix_dim, min_nb=min_nb)

            # Output
            args = (lyapex[0],)
            names = ('lyapex',)
            return utils.ReturnTuple(args, names)
