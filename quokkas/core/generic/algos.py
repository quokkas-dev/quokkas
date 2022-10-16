import pandas as pd
from ...utils.general_utils import maybe_quokkanize


def unique(values):
    return pd.unique(values)


def concat(*args, **kwargs):
    return maybe_quokkanize(pd.concat(*args, **kwargs))


def melt(df, *args, **kwargs):
    return df.melt(*args, **kwargs)


def merge(*args, **kwargs):
    return maybe_quokkanize(pd.merge(*args, **kwargs))


def to_datetime(*args, **kwargs):
    return maybe_quokkanize(pd.to_datetime(*args, **kwargs))
