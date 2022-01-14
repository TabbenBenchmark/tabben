"""
Provide integration support for autogluon (if installed) for the standard metrics.
"""

from functools import partial

from .metrics import ScikitLearnMetrics
from . import get_metrics as generic_get_metrics
from ..utils import has_package_installed

if not has_package_installed('autogluon'):
    raise ImportError('Please install autogluon to use the autogluon-specific things')
else:
    from autogluon.core.metrics import make_scorer


class AutogluonMetrics:
    mcc_binary = make_scorer(
        name='binary mcc',
        score_func=ScikitLearnMetrics.mcc_binary,
        optimum=1,
        greater_is_better=True
    )
    
    mcc_multiclass = make_scorer(
        name='multiclass mcc',
        score_func=ScikitLearnMetrics.mcc_multiclass,
        optimum=1,
        greater_is_better=True
    )
    
    auroc_binary = make_scorer(
        name='binary auroc',
        score_func=ScikitLearnMetrics.auroc_binary,
        optimum=1,
        greater_is_better=True
    )
    
    auroc_multiclass = make_scorer(
        name='multiclass auroc',
        score_func=ScikitLearnMetrics.auroc_multiclass,
        optimum=1,
        greater_is_better=True
    )
    
    ap_score = make_scorer(
        name='ap score',
        score_func=ScikitLearnMetrics.ap_score,
        optimum=1,
        greater_is_better=True
    )


# namespaced version of get_metrics function for autogluon-compatible metrics
def get_metrics(*args, **kwargs):
    """
    Return autogluon-compatible metrics given a dataset's task.

    See Also
    --------
    tabben.evaluators.get_metrics
    """
    return generic_get_metrics(*args, **kwargs, _namespace=AutogluonMetrics)
