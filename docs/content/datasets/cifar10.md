+++
title = "CIFAR10 Dataset"
+++

The CIFAR10 dataset is an image recognition dataset of low resolution images with 10 object classes, treated as a tabular dataset. Since each image is of a fixed size (32-by-32 with 3 color channels), we treat each pixel as if it is a tabular attribute. While an image model would likely work best, the idea is that the same tabular architecture is evaluated on a variety of datasets, including this one, and not evaluating image models. Another special feature of this dataset is that we use [*cleaned* test labels](https://github.com/cleanlab/label-errors) from [cleanlab](https://github.com/cleanlab/cleanlab). As we do *not* clean the training labels, this dataset is a noisy training data task, and this also makes our test set results more generalizable.

You can find out more about the original dataset from the [original website](https://www.cs.toronto.edu/~kriz/cifar.html). You can explore the cleaned labels from the [labelerrors.com](https://labelerrors.com/) website (select the CIFAR-10 dataset) and look into confident learning from their [paper](https://arxiv.org/abs/1911.00068).

## Data Preprocessing

We don't do much preprocessing to the data itself. The main change is the cleaned test labels (see above). We use the same official train-test split.
