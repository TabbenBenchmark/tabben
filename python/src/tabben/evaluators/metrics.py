from functools import partial

from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef

mcc = matthews_corrcoef

auroc_binary = roc_auc_score
auroc_multiclass = partial(roc_auc_score, multi_class='ovo')

ap_score = partial(average_precision_score, average='weighted')


def get_metrics(task, *, classes=2):
    if task == 'classification':
        return [auroc_binary if classes == 2 else auroc_multiclass, ap_score, mcc]
    elif task == 'regression':
        return []
    else:
        raise ValueError(f'Unsupported task `{task}`')
