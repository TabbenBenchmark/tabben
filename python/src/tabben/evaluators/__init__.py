from . import metrics


def get_metrics(task, *, classes=2, namespace=metrics):
    if task == 'classification':
        return [
            namespace.auroc_binary if classes == 2 else namespace.auroc_multiclass,
            namespace.ap_score,
            namespace.mcc
        ]
    elif task == 'regression':
        return []
    else:
        raise ValueError(f'Unsupported task `{task}`')

