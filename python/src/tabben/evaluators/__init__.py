"""
Implementations of the standard metrics used for the benchmark.
"""

from . import metrics


def get_metrics(task: str, *, classes: int = 2, namespace=metrics):
    if task == 'classification':
        return [
            namespace.auroc_binary if classes == 2 else namespace.auroc_multiclass,
            namespace.ap_score,
            namespace.mcc_binary if classes == 2 else namespace.mcc_multiclass,
        ]
    elif task == 'regression':
        return []
    else:
        raise ValueError(f'Unsupported task `{task}`')
