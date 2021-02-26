"""
Example showing the use of the mifs module.
"""

import FeatureSelection.mifs.mifs as mifs
from sklearn.datasets import make_classification, make_regression
import numpy as np
import matplotlib.pyplot as plt


def check_selection(selected, i, r):
    """
    Check FN, FP, TP ratios among the selected features.
    """
    # reorder selected features
    try:
        selected = set(selected)
        all_f = set(range(i+r))
        TP = len(selected.intersection(all_f))
        FP = len(selected - all_f)
        FN = len(all_f - selected)
        if (TP+FN) > 0:
            sens = TP/float(TP + FN)
        else:
            sens = np.nan
        if (TP+FP) > 0:
            prec = TP/float(TP + FP)
        else:
            prec = np.nan
    except:
        sens = np.nan
        prec = np.nan
    return sens, prec
    

if __name__ == '__main__':
    # variables for dataset    
    s = 200
    f = 10
    i = 5
    r = 0
    c = 2
    #
    # # simulate dataset with discrete class labels in y
    # X, y = make_classification(n_samples=s, n_features=f, n_informative=i, n_clusters_per_class=1,
    #                            n_redundant=r, n_classes=c, shuffle=False, random_state=1)
    #
    # # perform feature selection
    # MIFS = mifs.MutualInformationFeatureSelector(method='JMIM', verbose=2, n_features=10)
    # MIFS.fit(X,y)
    # # calculate precision and sensitivity
    # sens, prec = check_selection(np.where(MIFS._support_mask)[0], i, r)
    # print('Sensitivity: ' + str(sens) + '    Precision: ' + str(prec))
    #
    # plt.figure()
    # for i, idx in enumerate(MIFS.ranking_):
    #     plt.subplot(2, 5, i+1)
    #     # plt.title('Feature ' + str(i+1))
    #     plt.scatter(X[:, idx], y)
    #     plt.xlabel('Feature')
    #     plt.ylabel('Label')
    # plt.tight_layout()
    # plt.show()
    
    # simulate dataset with continuous y 
    X, y = make_regression(n_samples=s, n_features=f, n_informative=i,
                           random_state=0, shuffle=True, bias=-0.0)

    plt.figure(figsize=(13, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        # plt.title('Feature ' + str(i+1))
        plt.scatter(X[:, i], y, marker='.')
        plt.xlabel('Feature')
        plt.ylabel('Label')
    plt.show()

    # perform feature selection
    MIFS = mifs.MutualInformationFeatureSelector(method='JMIM', verbose=2,
                                                 categorical=False, n_features=3)
    MIFS.fit(X, y)
    # calculate precision and sensitivity
    sens, prec = check_selection(np.where(MIFS._support_mask)[0], i, r)
    print('Sensitivity: ' + str(sens) + '    Precision: ' + str(prec))

    plt.figure(figsize=(13, 4))
    for i, idx in enumerate(MIFS.ranking_):
        plt.subplot(1, 3, i+1)
        # plt.title('Feature ' + str(idx+1))
        plt.scatter(X[:, idx], y, marker='.')
        plt.xlabel('Feature')
        if i == 0:
            plt.ylabel('Label')
    plt.show()
