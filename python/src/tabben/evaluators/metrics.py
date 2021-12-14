"""
Implementations or imports of the standard metrics used for the benchmark
(scikit-learn-compatible and should work for most things working with numpy
arrays or numpy-like arrays).
"""

from functools import partial

from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef

mcc = matthews_corrcoef

auroc_binary = roc_auc_score
auroc_multiclass = partial(roc_auc_score, multi_class='ovo')

ap_score = partial(average_precision_score, average='weighted')

