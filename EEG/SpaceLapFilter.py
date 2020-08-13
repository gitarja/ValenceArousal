import numpy as np
class SpaceLapFilter:


    def FilterEEG(self, eeg, mode):
        surronds_elc8 = [[1, 11, 12], # Fp1, 1
                                [0, 11, 6, 16], # F7, 2
                              [7, 13, 14, 3], # F8, 3
                              [13, 2, 14, 15, 4], # T4, 4
                              [14, 3, 15, 18], # T6, 5
                              [6, 16, 9, 8], # T5, 6
                              [1, 11, 16, 5, 9], # T3, 7
                              [12, 13, 2], # Fp2, 8
                              [5, 9, 10], # O1, 9
                              [6, 16, 17, 5, 10, 8], # P3, 10
                              [16, 17, 14, 9, 15, 8, 18], # Pz, 11
                              [0, 1, 12, 6, 16, 17], # F3, 12
                              [0, 7, 11, 13, 16, 17, 14], # Fz, 13
                              [7, 12, 2, 17, 14, 3], # F4, 14
                              [12, 13, 2, 17, 3, 10, 15, 4], # C4, 15
                              [17, 14, 3, 10, 4, 18], # P4, 16
                              [1, 11, 12, 6, 17, 5, 9, 10], # C3, 17
                              [11, 12, 13, 16, 14, 9, 10, 15], # Cz, 18
                              [10, 15, 3]# O2, 19
                              ]
        surronds_elc4 = [[11], # Fp1, 1
                                [11, 6], # F7, 2
                              [13, 3], # F8, 3
                              [2, 14, 4], # T4, 4
                              [3, 15], # T6, 5
                              [6, 9], # T5, 6
                              [1, 16, 5], # T3, 7
                              [13], # Fp2, 8
                              [9], # O1, 9
                              [16, 5, 10, 8], # P3, 10
                              [17, 9, 15], # Pz, 11
                              [0, 1, 12, 16], # F3, 12
                              [11, 13, 17], # Fz, 13
                              [7, 12, 2, 14], # F4, 14
                              [13, 17, 3, 15], # C4, 15
                              [14, 10, 4, 18], # P4, 16
                              [11, 6, 17, 9], # C3, 17
                              [12, 16, 14, 10], # Cz, 18
                              [17]# O2, 19
                              ]
        filetered_signal = np.zeros_like(eeg)

        if mode == 4:
            for i in range(len(surronds_elc4)):
                ave = np.average(eeg[:, surronds_elc4[i]], axis=1)
                filetered_signal[:, i] = eeg[:, i] - ave
        elif mode == 8:
            for i in range(len(surronds_elc8)):
                ave = np.average(eeg[:, surronds_elc8[i]], axis=1)
                filetered_signal[:, i] = eeg[:, i] - ave

        return filetered_signal