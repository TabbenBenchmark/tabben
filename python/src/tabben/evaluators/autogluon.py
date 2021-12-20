"""
Provide integration support for autogluon (if installed) for the standard metrics.
"""

from collections import namedtuple
from functools import partial

from . import metrics
from . import get_metrics as generic_get_metrics
from ..utils import has_package_installed

if not has_package_installed('autogluon'):
    raise ImportError('Please install autogluon to use the autogluon-specific things')
else:
    from autogluon.core.metrics import make_scorer

mcc_binary = make_scorer(
    name='binary mcc',
    score_func=metrics.mcc_binary,
    optimum=1,
    greater_is_better=True
)

mcc_multiclass = make_scorer(
    name='multiclass mcc',
    score_func=metrics.mcc_multiclass,
    optimum=1,
    greater_is_better=True
)

auroc_binary = make_scorer(
    name='binary auroc',
    score_func=metrics.auroc_binary,
    optimum=1,
    greater_is_better=True
)

auroc_multiclass = make_scorer(
    name='multiclass auroc',
    score_func=metrics.auroc_multiclass,
    optimum=1,
    greater_is_better=True
)

ap_score = make_scorer(
    name='ap score',
    score_func=metrics.ap_score,
    optimum=1,
    greater_is_better=True
)

autogluon_globals = {
    key: val for key, val in globals().items()
    if not key.startswith('_')
}
attr_access_globals = namedtuple('autogluon', autogluon_globals.keys())(*autogluon_globals.values())

# namespaced version of get_metrics function for autogluon-compatible metrics
get_metrics = partial(generic_get_metrics, namespace=attr_access_globals)
