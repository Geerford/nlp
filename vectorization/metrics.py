import math

import numpy as np
import pandas as pd


def _get_cosine_similarity(a, b):
    numerator, denominator = sum([a[i] * b[i] for i in range(len(a))]), \
                             math.sqrt(sum([a[i] * a[i] for i in range(len(a))]) *
                                       sum([b[i] * b[i] for i in range(len(a))]))
    return numerator / denominator


def _get_weighted_jaccard_similarity(s, t):
    numerator, denominator = sum([min(s[i], t[i]) for i in range(len(s))]), \
                               sum([max(s[i], t[i]) for i in range(len(s))])
    return numerator / denominator


def _get_kullback_leibler_divergence(p, q, positive):
    with np.errstate(divide='ignore'):
        df = p.mul(np.log(p / q), fill_value=0)
        df[np.isinf(df)] = 0.0
    result = df.sum()
    if result == 0:
        return result
    if positive:
        return result
    else:
        return -result


def _get_jensen_shannon_divergence(p, q, positive):
    m = pd.Series((p + q) / 2)
    return _get_kullback_leibler_divergence(p, m, positive) / 2 + _get_kullback_leibler_divergence(q, m, positive) / 2


def get_cosine_similarity(coord, words, ppmi_df):
    i, j = coord
    return _get_cosine_similarity(ppmi_df[words[i]], ppmi_df[words[j]])


def get_weighted_jaccard_similarity(coord, words, ppmi_df):
    i, j = coord
    return _get_weighted_jaccard_similarity(ppmi_df[words[i]], ppmi_df[words[j]])


def get_kullback_leibler_divergence(coord, words, ppmi_df, positive=True):
    i, j = coord
    return _get_kullback_leibler_divergence(ppmi_df[words[i]], ppmi_df[words[j]], positive)


def get_jensen_shannon_divergence(coord, words, ppmi_df, positive=True):
    i, j = coord
    return _get_jensen_shannon_divergence(ppmi_df[words[i]], ppmi_df[words[j]], positive)
