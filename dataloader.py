"""
Data loading and preparation.
"""

import numpy as np


def load_data(image_path="data/images.npy", label_path="data/labels.npy"):
    """
    Loads images and labels from the given path, shuffles them pairwise
    and splits them into test, train, and validation sets
    """

    images = np.load(image_path)
    labels = np.load(label_path)

    assert len(images) == len(labels)

    p = np.random.permutation(len(labels))

    images = images[p]
    labels = labels[p]

    return split_data(images), split_data(labels)


def split_data(data, test_frac=.1, val_frac=.1):
    """
    Splits the given data array into three arrays according to the given fractions.
    Fractions are all in relation to the complete data set. Default is 10% test, 10% validation, 80% training
    Returns the tuple (training, validation, test)
    """

    rows = data.shape[0]
    test_end_idx = round(rows * test_frac)
    test = data[0: test_end_idx]

    val_end_idx = round(test_end_idx + rows * val_frac)
    val = data[test_end_idx + 1: val_end_idx]

    train = data[val_end_idx + 1: rows - 1]
    return train, val, test
