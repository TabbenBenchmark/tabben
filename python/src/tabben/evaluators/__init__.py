"""
Implementations of the standard metrics used for the benchmark.
"""

from . import metrics


def get_metrics(task: str, *, classes: int = 2, namespace=metrics):
    if task == 'classification':
        if classes == 2:
            return [namespace.auroc_binary, namespace.ap_score, namespace.mcc_binary]
        elif classes > 2:
            return [
                # namespace.auroc_multiclass,  # currently problematic
                namespace.mcc_multiclass,
            ]
        else:
            raise ValueError('invalid number of classes')
    elif task == 'regression':
        return []
    else:
        raise ValueError(f'Unsupported task `{task}`')
