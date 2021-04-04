from scipy.signal import butter, lfilter, convolve
import numpy as np
from datetime import datetime
from scipy import signal


def arWeight(ar):
    if ar == 0:
        return 1.2843347639484979
    else:
        return 0.8187414500683995


def valWeight(val):
    if val == 0:
        return 1.4341054313099042
    else:
        return 0.7676357417699872


def arToLabels(y):
    if (y < 3):
        return 0
    else:
        return 1


def valToLabels(y):
    if (y < 3):
        return 0

    else:
        return 1


def classifLabelsConv(y):
    if y == 0 or y == 1 or y == 2:
        return 0
    if y == 3:
        return 1
    if y == 4 or y == 5 or y == 6:
        return 2

    return y


def regressLabelsConv(y):
    if y == 0 or y == 1:
        return -2
    if y == 2:
        return -1
    if y == 3:
        return 0
    if y == 4:
        return 1
    if y == 5 or y == 6:
        return 2
    return y

def convertLabels(arousal, valence):
    if arousal == 0 and valence == 0:
        return 0
    elif arousal == 1 and valence == 0:
        return 1
    elif arousal == 0 and valence == 1:
        return 2
    else:
        return 3


def convertLabels(arousal, valence):
    if arousal == 0 and valence == 0:
        return 0
    elif arousal == 1 and valence == 0:
        return 1
    elif arousal == 0 and valence == 1:
        return 2
    else:
        return 3


def convertContrastiveLabels(time1, time2, sub1, sub2):
    if sub1 == sub2:
        if (time1 + 45 <= time2) or (time1 - 45 >= time2):
            return 0
        else:
            return 1
    else:
        return 1


def calcAccuracyRegression(y_ar, y_val, t_ar, t_val, th=0.5, mode="hard"):
    # labels
    B1 = (y_ar > 0) & (y_val > 0)
    A1 = (y_ar > 0) & (y_val < 0)
    B3 = (y_ar < 0) & (y_val > 0)
    A3 = (y_ar < 0) & (y_val < 0)
    A2 = (y_ar == 0) & (y_val < 0)
    B2 = (y_ar == 0) & (y_val > 0)
    C = y_val == 0
    if mode == "hard":
        B1_results = np.average((t_ar[B1] > 0.) & (t_val[B1] > 0))
        # ar positif and val negatif
        A1_results = np.average((y_val[A1] > 0) & (t_val[A1] < -0))
        # ar negatif and val positif
        B3_results = np.average((y_val[B3] < -0) & (t_val[B3] > 0))
        # ar negatif and val negatif
        A3_results = np.average((y_val[A3] < -0) & (t_val[A3] < -0))
        # val ambigous
        A2_results = np.average((np.abs(t_ar[A2]) <= th) & (t_val[A2] < 0))
        B2_results = np.average((np.abs(t_ar[B2]) <= th) & (t_val[B2] > 0))
    elif mode == "soft":
        B1_results = np.average((t_ar[B1] > -th) & (t_val[B1] > -th))
        # ar positif and val negatif
        A1_results = np.average((y_val[A1] > -th) & (t_val[A1] < th))
        # ar negatif and val positif
        B3_results = np.average((y_val[B3] < th) & (t_val[B3] > -th))
        # ar negatif and val negatif
        A3_results = np.average((y_val[A3] < th) & (t_val[A3] < th))
        # val ambigous
        A2_results = np.average((np.abs(t_ar[A2]) <= th) & (np.abs(t_val[A2]) <= th))
        B2_results = np.average((np.abs(t_ar[B2]) <= th) & (np.abs(t_val[B2]) <= 0))
    else:
        B1_results = np.average((t_val[B1] < 0) | ((t_ar[B1] < 0) & (t_val[B1] > 0)))
        # ar positif and val negatif
        A1_results = np.average((t_val[B1] > 0) | ((t_ar[B1] < 0) & (t_val[B1] < 0)))
        # ar negatif and val positif
        B3_results = np.average((t_val[B1] < 0) | ((t_ar[B1] > 0) & (t_val[B1] > 0)))
        # ar negatif and val negatif
        A3_results = np.average((t_val[B1] > 0) | ((t_ar[B1] > 0) & (t_val[B1] < 0)))
        # val ambigous
        A2_results = np.average(t_val[A2] > 0)
        B2_results = np.average(t_val[B2] < 0)

    # ar ambigous
    C_results = np.average(np.abs(t_ar[C]) <= th)

    template_1 = "{}, {}, {}, {}"
    template_2 = "{}, {}, {}"
    print("---------------Accuracy-----------------")
    print(template_1.format(B1_results, A1_results, B3_results, A3_results))
    print("--------------Accuracy Ambgigous-----------")
    print(template_2.format(A2_results, B2_results, C_results))


def windowFilter(x, numtaps=120, cutoff=2.0, fs=256.):
    b = signal.firwin(numtaps, cutoff, fs=fs, window='hamming', pass_zero='lowpass')
    y = lfilter(b, [1.0], x)
    return y


def utcToTimeStamp(x):
    utc = datetime.fromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')
    return timeToInt(utc)


def timeToInt(time):
    hours = time.split(" ")[-1]
    h, m, s = hours.split(":")
    inttime = 3600 * float(h) + 60 * float(m) + float(s)

    return inttime


def butterBandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butterBandpassFilter(data, lowcut, highcut, fs, order=5):
    b, a = butterBandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def avgSlidingWindow(x, n):
    '''
    Smooth a signal using a sliding window with n columns
    :param x: a signal
    :param n: number of columns
    :return: the smoothed signal
    '''

    window = np.ones(n) / n
    filtered = convolve(x, window, mode="same")

    return filtered


def rollingWindow(a, size=50):
    slides = []
    for i in range(len(a) // size):
        slides.append(a[(i * size):((i + 1) * size)])

    return np.array(slides)
