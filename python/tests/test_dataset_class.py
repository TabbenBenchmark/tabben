from itertools import permutations

from dataloading import check_split_sizes


def test_multiple_splits(data_dir):
    # this uses the covertype dataset, which has train, valid, test splits
    
    num_features = 54
    sizes = {
        'train': 11_340,
        'valid': 3_780,
        'test': 565_892,
    }

    check_split_sizes(
        data_dir, 'covertype', {
            (first, second): (first_size + second_size, num_features)
            for (first, first_size), (second, second_size) in permutations(sizes.items(), 2)
        }
    )
    