import biosppy
import nolds
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nn
import pyhrv.tools as tools
import pyhrv
from biosppy import utils
class ECGFeatures:

    def __init__(self, fs):
        self.fs = fs

    def extractRR(self, x):
        X, r = biosppy.signals.ecg.ecg(x, sampling_rate=self.fs, show=False)[1:3]
        r = biosppy.signals.ecg.correct_rpeaks(signal=X, rpeaks=r, sampling_rate=self.fs)[0]
        r = r.astype(float)
        # Compute NNI or RR
        nni = tools.nn_intervals(r)

        return nni
    def computeHeartBeat(self, x):
        ts, hb = biosppy.signals.ecg.ecg(x, sampling_rate=self.fs, show=False)[5:]
        return ts, hb

    def extractTimeDomain(self, x):
        nni = self.extractRR(x)
        nniParams = td.nni_parameters(nni=nni)
        nniSD = td.sdnn(nni=nni)
        nniDiff = td.nni_differences_parameters(nni=nni)
        nniDiffRM = td.rmssd(nni=nni)
        nniDiffSD = td.sdsd(nni=nni)
        hrParams = td.hr_parameters(nni=nni)

        return nniParams["nni_mean"], nniParams["nni_counter"],\
               nniSD["sdnn"],\
               nniDiff["nni_diff_mean"],  nniDiffRM["rmssd"],\
               nniDiffSD["sdsd"],\
               hrParams["hr_mean"], hrParams["hr_std"]

    def extractFrequencyDomain(self, x):
        nni = self.extractRR(x)
        #the bands was decided by refering to Revealing Real-Time Emotional Responses
        psd = fd.welch_psd(nni=nni, show=False,
                           fbands={'ulf': (0.00, 0.01), 'vlf': (0.01, 0.05), 'lf': (0.05, 0.15), 'hf': (0.15, 0.5)},
                           nfft=2 ** 12, legend=False, mode="dev")[0]

        return psd["fft_norm"][0], psd["fft_norm"][1], psd["fft_ratio"]

    def extractNonLinearDomain(self, x):
        nni = self.extractRR(x)
        sampEntro = nn.sample_entropy(nni=nni, dim=1)
        lyapEx = self.lyapunov_exponent(nni=nni, emb_dim=3, matrix_dim=2)

        return sampEntro["sampen"], lyapEx["lyapex"]

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
