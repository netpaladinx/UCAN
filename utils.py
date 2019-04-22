import numpy as np
import pandas as pd
import tensorflow as tf
import time


def get(dct, k):
    return dct.get(k, None) if isinstance(dct, dict) else None


def get_segment_ids(x):
    """ x: (np.array) d0 x 2, sorted
    """
    y = (x[1:] == x[:-1]).astype('uint8')
    return np.concatenate([np.array([0], dtype='int32'),
                           np.cumsum(1 - y[:, 0] * y[:, 1], dtype='int32')])

def get_unique(x):
    """ x: (np.array) d0 x 2, sorted
    """
    y = (x[1:] == x[:-1]).astype('uint8')
    return x[np.concatenate([np.array([1], dtype='bool'),
                             (1 - y[:, 0] * y[:, 1]).astype('bool')])]


def groupby_2cols_nlargest(x, y, k):
    """ x: (np.array) d0 x 2, sorted
        y: (np.array) d1
    """
    mask = (x[1:] == x[:-1]).astype('uint8')
    mask = (1 - mask[:, 0] * mask[:, 1]).astype('bool')
    n = len(x)
    key_idx = np.concatenate([np.array([0]), np.arange(1, n)[mask], np.array([n])])
    res_idx = np.concatenate([np.sort(s + np.argpartition(-y[s:e],
                                                          min(k - 1, e - s - 1))[:min(k, e - s)])
                              for s, e in zip(key_idx[:-1], key_idx[1:])])
    return res_idx.astype('int32')


def groupby_1cols_nlargest(x, y, k):
    """ x: (np.array) d0, sorted
        y: (np.array) d1
    """
    mask = (x[1:] != x[:-1])
    n = len(x)
    key_idx = np.concatenate([np.array([0]), np.arange(1, n)[mask], np.array([n])])
    res_idx = np.concatenate([np.sort(s + np.argpartition(-y[s:e],
                                                          min(k - 1, e - s - 1))[:min(k, e - s)])
                              for s, e in zip(key_idx[:-1], key_idx[1:])])
    return res_idx.astype('int32')


def groupby_1cols_merge(x, x_key, y_key, y_id):
    """ x: (np.array) d0, sorted
        x_key: (np.array): d0, unique in group
        y_key: (np.array): d1
        y_id: (np.array): d1
    """
    mask = (x[1:] != x[:-1])
    n = len(x)
    key_idx = np.concatenate([np.array([0]), np.arange(1, n)[mask], np.array([n])])
    yid_li = [y_id[np.in1d(y_key, x_key[s:e])]
              for s, e in zip(key_idx[:-1], key_idx[1:])]
    res_idx = np.concatenate(yid_li)
    grp_idx = np.concatenate([np.repeat(np.array([i], dtype='int32'), len(yid)) for i, yid in enumerate(yid_li)])
    return res_idx, grp_idx

