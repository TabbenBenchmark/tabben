"""
Implementations of the standard metrics used for the benchmark.
"""

from . import metrics

__all__ = [
    'get_metrics'
]


def get_metrics(task: str, *, classes: int = 2, _namespace=metrics.ScikitLearnMetrics):
    """
    Return the specific metric implementations given a dataset's task (and number
    of classes if classification).
    
    Parameters
    ----------
    task : str
        Task of the dataset
    classes : int, default=2
        Number of classes if the dataset has a classification task
    _namespace
        Namespace containing the specific implementations of the metrics
        (this is an internal parameter that is used to switch backends)

    Returns
    -------
    sequence of callable
        Set of metrics/functions that should be used to evaluate models on this benchmark
    """
    
    if task == 'classification':
        if classes == 2:
            return [_namespace.auroc_binary, _namespace.ap_score, _namespace.mcc_binary]
        elif classes > 2:
            return [
                # namespace.auroc_multiclass,  # currently problematic
                _namespace.mcc_multiclass,
            ]
        else:
            raise ValueError('invalid number of classes')
    elif task == 'regression':
        return []
    else:
        raise ValueError(f'Unsupported task `{task}`')
