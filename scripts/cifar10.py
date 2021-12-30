"""
Converts the CIFAR10 dataset (using the torchvision dataset) to the standard dataset
format used for this benchmark. Each pixel of an image is treated as a tabular
attribute.

The test labels use the cleaned test labels from cleanlab. First download the cross-
validated predicted labels from:
https://github.com/cleanlab/label-errors/raw/main/cross_validated_predicted_labels
/cifar10_test_set_pyx_argmax_predicted_labels.npy
and put in the source directory. This means that models are trained using noisy
labels but evaluated on relatively clean labels.
"""
import os

import numpy as np
from torchvision.datasets import CIFAR10

from utils import default_config, save_json, save_npz

bibtex = """\
@article{krizhevsky_learning_nodate,
    title = {Learning Multiple Layers of Features from Tiny Images},
    pages = {60},
    author = {Krizhevsky, Alex},
    langid = {english},
    keywords = {cmsc673},
}

@article{northcutt2021confidentlearning,
    title={Confident Learning: Estimating Uncertainty in Dataset Labels},
    author={Curtis G. Northcutt and Lu Jiang and Isaac L. Chuang},
    journal={Journal of Artificial Intelligence Research (JAIR)},
    volume={70},
    pages={1373--1411},
    year={2021}
}"""

label_categories = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]


def extract_arrays(split, config):
    dataset = CIFAR10(config.source, train=split == 'train', download=True)
    
    # preallocate the array
    all_inputs = np.empty((len(dataset), 32 * 32 * 3), dtype=np.float32)
    all_outputs = np.empty(len(dataset), np.int_)
    
    for index, (input, output) in enumerate(dataset):
        all_inputs[index, :] = np.array(input, dtype=np.float32).reshape(32 * 32 * 3) / 255
        all_outputs[index] = output
    
    return all_inputs, all_outputs


def save_arrays(config):
    X_train, y_train = extract_arrays('train', config)
    X_test, y_test = extract_arrays('test', config)
    
    y_test_clean = np.load(os.path.join(config.source, 'cifar10_test_set_pyx_argmax_predicted_labels.npy'))
    
    print(f'Replaced {100 * (y_test != y_test_clean).sum() / 10_000:.2f}% of test labels')
    
    if config.dataset_file:
        save_npz(
            os.path.join(config.outputdirectory, 'cifar10'), {
                'train-data': X_train,
                'train-labels': y_train,
                'test-data': X_test,
                'test-labels': y_test_clean,
                '_columns-data': np.array([f'P{i}' for i in range(32 * 32 * 3)], dtype=np.str_),
                '_columns-labels': np.array(['label'], dtype=np.str_),
            }
        )
    
    if config.extras_file:
        save_json(
            {
                'categories': {
                    'label': label_categories,
                },
                'bibtex': bibtex,
            }, os.path.join(config.outputdirectory, 'cifar10.json')
        )


if __name__ == '__main__':
    config = default_config()
    save_arrays(config)
