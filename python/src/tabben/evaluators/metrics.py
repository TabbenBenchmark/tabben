"""
Implementations or imports of the standard metrics used for the benchmark
(scikit-learn-compatible and should work for most things working with numpy
arrays or numpy-like arrays).
"""

from functools import partial

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef


class ScikitLearnMetrics:
    ap_score = partial(average_precision_score, average='weighted')

    auroc_binary = roc_auc_score
    auroc_multiclass = partial(roc_auc_score, multi_class='ovr')
    
    @staticmethod
    def mcc_binary(y_true, y_pred):
        sample_points = 15
        mcc_sum = 0
    
        for threshold in np.linspace(0, 1, num=sample_points):
            mcc_sum += matthews_corrcoef(y_true, y_pred > threshold)
    
        return mcc_sum / sample_points

    @staticmethod
    def mcc_multiclass(y_true, y_pred):
        sample_points = 15
        mcc_sum = 0
    
        for threshold in np.linspace(0, 1, num=sample_points):
            mcc_sum += matthews_corrcoef(y_true, y_pred.argmax(axis=1))
    
        return mcc_sum / sample_points

