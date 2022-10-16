import numpy as np
import pandas as pd

from .sk_utils import is_scalar_nan


def approximately_equal(df, other, error=1e-5, skipna=False):
    """
    Returns true if df is approximately equal to other

    :param df: first object to be compared
    :param other: second object to be compared
    :param error: allowed size of discrepancies
    :param skipna: if true, the na-s in df won't affect the result
    :return: True if approximately equal, else False
    """

    if not skipna:
        return ((df - other).abs() < error).all().all()
    else:
        return (((df - other).abs() < error) | df.isna()).all().all()


def approximately_equal_scalars(a, b, error=1, equal=False):
    """
    Returns true if two scalars are approximately equals

    :param a: first scalar
    :param b: second scalar
    :param error: allowed absolute difference
    :param equal: if the inequality can be non-strict
    :return: True if approximately equal, else False
    """
    if equal:
        return abs(a - b) <= error
    else:
        return abs(a - b) < error


def isnan_safe_1d(series):
    """
    Returns isna result for 1d argument.
    Is safe to use with object series / ndarrays

    :param series: series or ndarray to be checked
    :return: boolean series or ndarray if is nan
    """
    if series.dtype.kind in 'OUS':
        nd = series.to_numpy()
        arr = []
        for i in range (nd.shape[0]):
            arr.append(is_scalar_nan(nd[i]))
        return pd.Series(arr, index=series.index)
    else:
        return np.isnan(series)


def series_is_in(left, right):
    """
    Returns True if all "True" values in left are also included in right.

    :param left: first dataframe / ndarray
    :param right: seconda dataframe / ndarray
    :return: True if in, False if not
    """
    return ((left & right) == left).all()
