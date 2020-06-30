from scipy.signal import butter, lfilter, convolve
import numpy as np
class Utils:

    # timestampt to int
    def timeToInt(self, time):
        date, hours = time.split(" ")
        h, m, s = hours.split(":")
        inttime = 3600 * float(h) + 60 * float(m) + float(s)

        return inttime

    def butterBandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butterBandpassFilter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butterBandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def avgSlidingWindow(self, x, n):
        '''
        Smooth a signal using a sliding window with n columns
        :param x: a signal
        :param n: number of columns
        :return: the smoothed signal
        '''

        window = np.ones(n) / n
        filtered = convolve(x, window, mode="same")

        return filtered