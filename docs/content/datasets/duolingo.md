+++
title = "Duolingo Dataset"
+++

The Duolingo space repetition learning dataset is a set of learning traces from the Duolingo app (anonymized). This is the largest dataset in the benchmark in terms of number of examples. Note that the implicit structure within this dataset is expected to be different from many other tabular datasets, so it may be a good idea to check out the Duolingo spaced repetition model [paper](https://github.com/duolingo/halflife-regression/blob/master/settles.acl16.pdf) for some theory or [python implementation](https://github.com/duolingo/halflife-regression/blob/master/experiment.py).

We use 2 different Duolingo datasets, referenced as `duolingo-original` and `duolingo-categorical`. The `duolingo-original` uses the features originally used in the original Duolingo spaced learning experiments (mostly, it excludes the lexeme information and only includes the lexeme id). The `duolingo-categorical` partially uses the lexeme information in a naive way, mapping different elements of the lexeme string into individual categorical attributes and excludes the redundant lexeme_id.

You can find out more about the original dataset at the [Duolingo halflife regression GitHub](https://github.com/duolingo/halflife-regression#data-set-and-format).

## Data Preprocessing

Both variants of the dataset have the same custom 80-20 train-test split. Splits are created so that no user_id occurs in both splits.

We don't do any further processing on the `duolingo-original` variant of the dataset. For the `duolingo-categorical` variant, we split and process the tags in the lexeme_string into a sparsely encoded representation.
