"""
Creates a variety of different fake datasets useful for basic testing.
"""

import itertools

import numpy as np
import pandas as pd


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def create_test_dataset(m, n, o, c):
    input_attributes = pd.DataFrame(np.random.randn(m, n))
    input_attributes.columns = [f'A{i}' for i in range(n)]
    
    labels = pd.DataFrame(np.random.randint(0, c, (m, o)))
    labels.columns = [f'label{i}' for i in range(o)]
    
    return pd.concat([input_attributes, labels], axis=1)


def split_dataset(df, splits):
    m = df.shape[0]
    assert sum(splits) == 1 and all(0 < split <= 1 for split in splits)
    
    sizes = [0] + [split_ratio * m for split_ratio in splits]
    if sum(sizes) > m:
        sizes[-1] = m - sum(sizes[:-1])
    
    subsets = [df.iloc[start:stop] for start, stop in pairwise(itertools.accumulate(sizes))]
    
    return subsets

