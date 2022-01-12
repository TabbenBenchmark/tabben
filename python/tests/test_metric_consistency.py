import itertools

import numpy as np
import pytest
from numpy.random import default_rng

from tabben.evaluators.metrics import ScikitLearnMetrics as metrics
from tabben.evaluators.autogluon import AutogluonMetrics as ag_metrics


################################################################################
# Common Utilities
################################################################################

@pytest.fixture(params=[10, 50, 100, 1000])
def binary_classification_outputs(request):
    num_examples = request.param
    
    rng = default_rng()
    y_true = rng.integers(0, 1 + 1, size=(num_examples,))
    y_pred = rng.random(size=(num_examples,))
    return y_true, y_pred


@pytest.fixture(
    params=list(
        itertools.product(
            [10, 50, 100, 1000],  # number of examples
            [3, 5, 16, 17]     # number of classes
        )
    )
)
def multi_classification_outputs(request):
    num_examples, num_classes = request.param
    
    rng = default_rng()
    y_true = rng.integers(0, num_classes, size=(num_examples,))
    y_pred = rng.random(size=(num_examples, num_classes))
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    return y_true, y_pred


################################################################################
# Binary classification metrics
################################################################################

def test_auroc_binary(binary_classification_outputs):
    np.testing.assert_allclose(
        metrics.auroc_binary(*binary_classification_outputs),
        ag_metrics.auroc_binary(*binary_classification_outputs)
    )


def test_ap_score_binary(binary_classification_outputs):
    np.testing.assert_allclose(
        metrics.ap_score(*binary_classification_outputs),
        ag_metrics.ap_score(*binary_classification_outputs)
    )


def test_mcc_binary(binary_classification_outputs):
    np.testing.assert_allclose(
        metrics.mcc_binary(*binary_classification_outputs),
        ag_metrics.mcc_binary(*binary_classification_outputs)
    )


################################################################################
# Multiclass classification metrics
################################################################################

@pytest.mark.skip(reason='multiclass auroc is problematic')
def test_auroc_multiclass(multi_classification_outputs):
    print(multi_classification_outputs[0].shape, multi_classification_outputs[1].shape)
    np.testing.assert_allclose(
        metrics.auroc_multiclass(*multi_classification_outputs),
        ag_metrics.auroc_multiclass(*multi_classification_outputs)
    )
